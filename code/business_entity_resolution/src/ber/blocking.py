"""Candidate generation: S2/S3 record -> S1 entities, always within country.

Passes (see PLAN.md §4):
  B1  exact address key 'house number|next informative token'
  B2  char 3-gram TF-IDF cosine on the core name, top-k within (country, region);
      records without a region search the whole country, and every record also
      searches the country's pool of S1 entities without a (real) region.
  B6  the same search with word TF-IDF on the address minus its region parts
      (catches native-script and unrelated names at the same address).
  B5  the same search with the fine-tuned multilingual name encoder
      (ber.embed), when embeddings exist: cross-script names.

Regions: S1 via geo.assign_region. S2/S3 via a lexicon learned from
*pseudo-links* (B1 pairs whose names nearly agree), so no labels are needed
and it works for a country never seen in training (France).

Frames: s1 has s1_id; m (S2+S3 stacked) has m_src, m_id; both carry the
normalised columns from ber.normalize plus `country`.
"""
import os

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer

from .geo import assign_region, assign_region_by_lexicon, learn_region_lexicon, part_frequencies
from .topk import rowwise_dot, topk

B1_MAX_BLOCK = 20        # skip address keys shared by more S1 entities
PSEUDO_MIN_NAME = 90     # token_set_ratio for a B1 pair to count as a pseudo-link
MIN_REGION_SHARE = 0.005 # smaller S1 "regions" ('1st floor', a city) join the pool
K_REGION = 10
K_POOL = 3
MIN_COS = 0.1
TFIDF = dict(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, min_df=2,
             max_df=0.05, dtype=np.float32)
TFIDF_ADDR = dict(analyzer="word", token_pattern=r"\S+", sublinear_tf=True, min_df=2,
                  max_df=0.05, dtype=np.float32)
K_ADDR = 5
K_EMB = 10
MIN_ECOS = 0.3
# Search deeper in large regions: top-k lists fill with look-alikes when a region has
# 100k+ entities (Maharashtra, Delhi). k grows with sqrt(region size / REGION_K_REF),
# capped at K_SCALE_MAX x the base k. Learned pruning later removes the extra junk.
K_SCALE = os.environ.get("BER_K_SCALE", "0") == "1"   # experiment: BER_K_SCALE=1
REGION_K_REF = 30_000
K_SCALE_MAX = 3


def scaled_k(k, n_index):
    if not K_SCALE:
        return k
    return int(min(k * K_SCALE_MAX, max(k, round(k * (n_index / REGION_K_REF) ** 0.5))))
VEC_FIT_MAX = 3_000_000


def s1_regions(s1):
    """Add `region`: the most frequent address part that is a real region.

    A first pass finds each record's most frequent part; parts chosen by at
    least MIN_REGION_SHARE of the country are the real regions. Records whose
    top part was a pseudo-region ('1st floor', a big city) fall back to their
    best real-region part; records with none form the pool ('').
    """
    s1a = s1.with_columns(pl.col("addr_norm").alias("address"))
    freqs = part_frequencies(s1a)
    first = assign_region(s1a, freqs)
    real = (first.group_by("country", "region").len("n")
            .filter((pl.col("region") != "")
                    & (pl.col("n") / pl.col("n").sum().over("country") >= MIN_REGION_SHARE))
            .select("country", pl.col("region").alias("part")))
    return assign_region(s1a, freqs.join(real, on=["country", "part"], how="semi"), min_df=0).drop("address")


def b1_pairs(s1, m):
    k1 = (s1.select("s1_id", "country", pl.col("addr_keys").alias("key")).explode("key")
          .drop_nulls("key").unique())
    k1 = k1.filter(pl.len().over("country", "key") <= B1_MAX_BLOCK)
    km = (m.select("m_src", "m_id", "country", pl.col("addr_keys").alias("key")).explode("key")
          .drop_nulls("key").unique())
    return (km.join(k1, on=["country", "key"]).select("s1_id", "m_src", "m_id").unique())


def name_scores(pairs, s1, m, scorer=fuzz.token_set_ratio):
    """Vector of scorer(s1 name_core, m name_core) aligned with `pairs`."""
    x = (pairs.join(s1.select("s1_id", pl.col("name_core").alias("a")), on="s1_id", how="left")
         .join(m.select("m_src", "m_id", pl.col("name_core").alias("b")), on=["m_src", "m_id"], how="left"))
    return process.cpdist(x["a"].to_list(), x["b"].to_list(), scorer=scorer, workers=-1)


def region_lexicon(b1, s1, m):
    """Lexicon noisy-address-part -> S1 region, learned from pseudo-links."""
    score = name_scores(b1, s1, m)
    links = b1.with_columns(pl.Series("score", score)).filter(pl.col("score") >= PSEUDO_MIN_NAME)
    links = links.filter(pl.len().over("m_src", "m_id") == 1)        # unambiguous owner only
    linked = (links.join(s1.select("s1_id", "region"), on="s1_id")
              .filter(pl.col("region") != "")
              .join(m.select("m_src", "m_id", "country", pl.col("addr_norm").alias("address")),
                    on=["m_src", "m_id"]))
    lex = learn_region_lexicon(linked.select("country", "address", "region"))
    ident = (s1.filter(pl.col("region") != "").group_by("country", "region").len("support")
             .select("country", pl.col("region").alias("part"), "region",
                     (pl.col("support") * 1000).cast(pl.UInt32).alias("support")))
    lex = pl.concat([ident, lex.cast({"support": pl.UInt32})]).unique(["country", "part"], keep="first")
    return lex, links.height


def m_regions(m, lexicon):
    out = assign_region_by_lexicon(
        m.select("m_src", "m_id", "country", pl.col("addr_norm").alias("address")), lexicon)
    return m.with_columns(out["region"])


def addr_noregion(df, lexicon):
    """Address without the parts that name a region (state-format noise)."""
    parts = (df.select(pl.int_range(pl.len(), dtype=pl.UInt32).alias("row_idx"), "country",
                       pl.col("addr_norm").str.split(", ").alias("part"))
             .explode("part")
             .join(lexicon.select("country", "part"), on=["country", "part"], how="anti")
             .group_by("row_idx").agg(pl.col("part").str.join(" ").alias("addr_nr")))
    return (df.with_row_index("row_idx").join(parts, on="row_idx", how="left").drop("row_idx")
            .with_columns(pl.col("addr_nr").fill_null("")))


def _fit_vec(texts, params):
    if len(texts) > VEC_FIT_MAX:
        rng = np.random.default_rng(0)
        texts = [texts[i] for i in rng.choice(len(texts), VEC_FIT_MAX, replace=False)]
    return TfidfVectorizer(**params).fit(texts)


def _tfidf_pass(X, Q, s1_reg, m_reg, k_region, k_pool, log=None):
    """Top-k TF-IDF search within region (+ country-wide for region-less, + pool)."""
    return _region_pass(lambda qi, xi, k: topk(Q[qi], X[xi], k, min_score=MIN_COS),
                        X.shape[0], s1_reg, m_reg, k_region, k_pool, log)


def _region_pass(search, n_x, s1_reg, m_reg, k_region, k_pool, log=None):
    """Run search(query_idx, index_idx, k) -> (r, c, score) per region -> q, x, rank."""
    out = []

    def add(qi, xi, k, what):
        if len(qi) == 0 or len(xi) == 0:
            return
        if what != "pool":
            k = scaled_k(k, len(xi))
        if log is not None:
            log(f"    {what}: {len(qi):,} queries x {len(xi):,} S1 (k={k})")
        r, c, v = search(qi, xi, k)
        df = pl.DataFrame({"q": qi[r].astype(np.int64), "x": xi[c].astype(np.int64), "cos": v})
        out.append(df.with_columns(  # rank within this search, 0 = best
            pl.col("cos").rank("ordinal", descending=True).over("q").cast(pl.Int16).sub(1).alias("rank")))

    for reg in np.unique(s1_reg):
        if reg != "":
            add(np.flatnonzero(m_reg == reg), np.flatnonzero(s1_reg == reg), k_region, reg)
    add(np.flatnonzero(m_reg == ""), np.arange(n_x), k_region, "no region")
    add(np.flatnonzero(m_reg != ""), np.flatnonzero(s1_reg == ""), k_pool, "pool")
    if not out:
        return pl.DataFrame(schema={"q": pl.Int64, "x": pl.Int64, "rank": pl.Int16})
    return (pl.concat(out).sort("rank").unique(["q", "x"], keep="first").select("q", "x", "rank"))


def _chunked_dot(A, B, xi, qi, chunk=2_000_000):
    return np.concatenate([rowwise_dot(A[xi[i:i + chunk]], B[qi[i:i + chunk]])
                           for i in range(0, len(xi), chunk)]) if len(xi) else np.empty(0, np.float32)


def _gather_emb(split, df, src_col=None):
    """Name embeddings for the rows of df (column `row` = row in the record parquet)."""
    from .embed import emb_path, load_embeddings
    srcs = (1,) if src_col is None else (2, 3)
    if not all(emb_path(split, s).exists() for s in srcs):
        return None
    out = None
    for s in srcs:
        sel = (np.arange(df.height) if src_col is None
               else np.flatnonzero(df[src_col].to_numpy() == s))
        rows = df["row"].to_numpy()[sel]
        order = np.argsort(rows)                     # sorted reads from the memmap
        e = np.asarray(load_embeddings(split, s)[rows[order]])
        if out is None:
            out = np.empty((df.height, e.shape[1]), dtype=np.float16)
        out[sel[order]] = e
    return out


def _country(s1c, mc, b1, log, split=None):
    s1_reg, m_reg = s1c["region"].to_numpy(), mc["region"].to_numpy()
    vn = _fit_vec(s1c["name_core"].to_list() + mc["name_core"].to_list(), TFIDF)
    Xn, Qn = vn.transform(s1c["name_core"].to_list()), vn.transform(mc["name_core"].to_list())
    log(f"  {s1c['country'][0]}: name pass")
    b2 = _tfidf_pass(Xn, Qn, s1_reg, m_reg, K_REGION, K_POOL, log)
    va = _fit_vec(s1c["addr_nr"].to_list() + mc["addr_nr"].to_list(), TFIDF_ADDR)
    Xa, Qa = va.transform(s1c["addr_nr"].to_list()), va.transform(mc["addr_nr"].to_list())
    log(f"  {s1c['country'][0]}: address pass")
    b6 = _tfidf_pass(Xa, Qa, s1_reg, m_reg, K_ADDR, K_POOL, log)
    E1 = Em = None
    if split is not None and "row" in s1c.columns and not os.environ.get("BER_NO_EMB"):
        E1, Em = _gather_emb(split, s1c), _gather_emb(split, mc, "m_src")
    if E1 is not None and Em is not None:
        from .embed import gpu_dense_topk
        log(f"  {s1c['country'][0]}: embedding pass")
        b5 = _region_pass(lambda qi, xi, k: gpu_dense_topk(Em[qi], E1[xi], k, min_score=MIN_ECOS),
                          len(E1), s1_reg, m_reg, K_EMB, K_POOL, log)
    else:
        b5 = pl.DataFrame(schema={"q": pl.Int64, "x": pl.Int64, "rank": pl.Int16})
    log(f"  {s1c['country'][0]}: {len(np.unique(s1_reg))} regions, "
        f"{(m_reg == '').sum():,} region-less records, pool {(s1_reg == '').sum():,} S1; "
        f"B2 {b2.height:,} B6 {b6.height:,} B5 {b5.height:,}")

    s1_idx = s1c.select("s1_id", pl.int_range(pl.len(), dtype=pl.Int64).alias("x"))
    m_idx = mc.select("m_src", "m_id", pl.int_range(pl.len(), dtype=pl.Int64).alias("q"))
    b1c = (b1.join(s1_idx, on="s1_id").join(m_idx, on=["m_src", "m_id"])
           .select("q", "x", pl.lit(True).alias("b1")))
    both = (b2.rename({"rank": "rank_name"})
            .join(b6.rename({"rank": "rank_addr"}), on=["q", "x"], how="full", coalesce=True)
            .join(b5.rename({"rank": "rank_emb"}), on=["q", "x"], how="full", coalesce=True)
            .join(b1c, on=["q", "x"], how="full", coalesce=True))
    xi, qi = both["x"].to_numpy(), both["q"].to_numpy()
    both = both.with_columns(
        pl.Series("cos", _chunked_dot(Xn, Qn, xi, qi)),
        pl.Series("acos", _chunked_dot(Xa, Qa, xi, qi)),
        pl.col("b1").fill_null(False),
        pl.col("rank_name").fill_null(99), pl.col("rank_addr").fill_null(99),
        pl.col("rank_emb").fill_null(99))
    cols = ["s1_id", "m_src", "m_id", "cos", "acos", "rank_name", "rank_addr", "b1"]
    if E1 is not None and Em is not None:
        from .embed import rowwise_cos
        both = both.with_columns(pl.Series("ecos", rowwise_cos(E1, Em, xi, qi)))
        cols += ["ecos", "rank_emb"]
    return both.join(s1_idx, on="x").join(m_idx, on="q").select(cols)


def generate(s1, m, log=print, sink=None, split=None):
    """Return (candidates, s1 with region/addr_nr, m with region/addr_nr, lexicon).

    With `sink`, each country's candidates go to sink(country, frame) instead of
    being concatenated (candidates is then None), bounding memory on test.
    """
    s1 = s1_regions(s1)
    b1 = b1_pairs(s1, m)
    lexicon, n_links = region_lexicon(b1, s1, m)
    m = m_regions(m, lexicon)
    s1, m = addr_noregion(s1, lexicon), addr_noregion(m, lexicon)
    log(f"B1 pairs {b1.height:,}; pseudo-links {n_links:,}; lexicon {lexicon.height:,} parts; "
        f"records with region {(m['region'] != '').mean():.1%}")
    cands = []
    for country in sorted(s1["country"].unique().to_list()):
        s1c = s1.filter(pl.col("country") == country)
        mc = m.filter(pl.col("country") == country)
        if mc.height == 0:
            continue
        c = _country(s1c, mc, b1, log, split).with_columns(pl.col("m_src").cast(pl.Int8))
        log(f"  {country}: {c.height:,} candidates")
        if sink is None:
            cands.append(c)
        else:
            sink(country, c)
        del c, s1c, mc
    return (pl.concat(cands) if sink is None else None), s1, m, lexicon

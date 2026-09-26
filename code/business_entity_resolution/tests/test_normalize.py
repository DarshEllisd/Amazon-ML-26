import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber.normalize import normalize_address, normalize_name  # noqa: E402


def test_name_junk_legal_and_brackets():
    assert normalize_name("VISION SYSTEMS PRIVATE [LIMITED]") == ("vision systems", "ltd pvt", False)
    assert normalize_name("*** NO Worldwide")[0] == normalize_name("NO Worldwide Ínc")[0] == "no worldwide"
    assert normalize_name("Pacific Consultants L.L.C.")[1] == "llc"


def test_name_domain_leet_duplicates():
    assert normalize_name("howiedotyfairclough.com")[0] == "howiedotyfairclough"
    assert normalize_name("M0DERN BANC5HARES")[0] == "modern bancshares"
    assert normalize_name("Vapi Vapi Direct Pvt") == ("vapi direct", "pvt", False)
    assert normalize_name("Laborers Local Union 489")[0] == "laborers local union 489"


def test_name_native_script_is_transliterated():
    core, legal, native = normalize_name("ভিশন সিস্টেমস প্রাইভেট লিমিটেড")
    assert native and core.isascii() and legal == "ltd"


def test_name_only_legal_words_keeps_something():
    assert normalize_name("Inc")[0] == "inc"


def test_address_numbers_and_keys():
    norm, keys = normalize_address("#3550 NM 528, ALBUQUERQUE, NM")
    assert norm == "3550 nm 528, albuquerque, nm" and keys[0] == "3550|nm"
    assert normalize_address("003778 N Baxter Ave") == ("3778 n baxter ave", ["3778|baxter"])
    # '1/236' and 'No /236' share the key 236|xavier
    assert "236|xavier" in normalize_address("1/236 Xavier Bhavan, Kanj")[1]
    assert "236|xavier" in normalize_address("No /236 Xavier Bhavan, Kanyakumari, TN")[1]
    # ordinals canonicalised, street types abbreviated
    assert normalize_address("1165 38st Street, X")[0] == normalize_address("1165 38th St, X")[0]
    assert normalize_address("134-B Rue de Lille, Lille")[1] == ["134|lille"]
    assert normalize_address("") == ("", [])

from avl.config import Brand
from avl.extract import BrandMatcher, find_domains


def test_brand_whole_word_only():
    m = BrandMatcher([Brand("Noon"), Brand("Etsy")])
    assert m.find("Shop on Noon or etsy today") == ["Noon", "Etsy"]
    assert m.find("Come back in the afternoon") == []


def test_brand_alias_returns_canonical_name():
    m = BrandMatcher([Brand("Personalization Mall", aliases=("PersonalizationMall",))])
    assert m.find("Try PersonalizationMall.") == ["Personalization Mall"]


def test_domains_lowercase_no_www_deduplicated():
    text = "See https://www.Etsy.com/uk and etsy.com, also zazzle.ae."
    assert find_domains(text) == ["etsy.com", "zazzle.ae"]


def test_no_domains():
    assert find_domains("No links here, e.g. nothing.") == []

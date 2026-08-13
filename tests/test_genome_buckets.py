from audience_predictor.data.genome_buckets import bucket_tag, partition_tags


def test_script_visible_tropes():
    for tag in (
        "dystopia",
        "twist ending",
        "time travel",
        "claustrophobic",
        "anti-hero",
        "found footage",
        "slow paced",
        "horror",
        "witty",
        "18th century",
        "space opera",
    ):
        assert bucket_tag(tag) == "script", tag


def test_reception_awards_and_quality():
    for tag in (
        "oscar",
        "oscar winner",
        "oscar (best picture)",
        "masterpiece",
        "overrated",
        "imdb top 250",
        "great acting",
        "bad cgi",
        "cult classic",
        "afi 100",
        "golden palm",
    ):
        assert bucket_tag(tag) == "reception", tag


def test_production_format_franchise_talent():
    for tag in (
        "imax",
        "cgi",
        "pixar",
        "star wars",
        "sequel",
        "based on a book",
        "hitchcock",
        "007",
        "franchise",
        "pg-13",
        "author:neil gaiman",
    ):
        assert bucket_tag(tag) == "production", tag


def test_good_versus_evil_stays_script():
    assert bucket_tag("good versus evil") == "script"


def test_classic_car_stays_script():
    assert bucket_tag("classic car") == "script"


def test_partition_covers_every_tag():
    names = ["horror", "oscar", "imax", "twist ending", "masterpiece"]
    parts = partition_tags(names)
    assert set(sum(parts.values(), [])) == set(names)
    assert "horror" in parts["script"]
    assert "oscar" in parts["reception"]
    assert "imax" in parts["production"]

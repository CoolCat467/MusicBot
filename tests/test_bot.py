from musicbot.bot import combine_end


def test_combine_end() -> None:
    assert combine_end(("one",)) == "one"
    assert combine_end(("one", "two")) == "one and two"
    assert combine_end(("one", "two", "three")) == "one, two, and three"

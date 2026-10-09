from dhwani.signatures import SignatureDB


def test_save_and_list_roundtrip(tmp_path):
    """2.D: the library saves and lists; there is no matcher to promise."""
    db = SignatureDB(str(tmp_path / "s.sqlite"))
    sid = db.save_from_result({"modulation": {"prediction": "QPSK"},
                               "frames": {"sync_word_hex": "1acffc1d"}}, "x")
    [row] = db.list_all()
    assert (row["id"], row["name"], row["modulation"], row["sync_word_hex"]) \
        == (sid, "x", "QPSK", "1acffc1d")
    assert not hasattr(db, "match")
    db.close()

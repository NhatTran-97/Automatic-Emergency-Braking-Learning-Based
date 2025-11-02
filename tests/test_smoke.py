def test_import():
    import app.main as m
    demo = m.launch()
    assert demo is not None
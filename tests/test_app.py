from app import app

def test_root_status_code():
    #test that just checks the home page responds
    test_client = app.test_client()
    resp = test_client.get("/")
    assert resp.status_code == 200


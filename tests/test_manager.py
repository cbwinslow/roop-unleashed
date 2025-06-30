from agents import MultiAgentManager


def test_assist_unknown():
    mgr = MultiAgentManager()
    assert 'Unknown agent' == mgr.assist('bogus', 'hello')


def test_assist_install():
    mgr = MultiAgentManager()
    resp = mgr.assist('installer', 'how to install?')
    assert 'pip install' in resp

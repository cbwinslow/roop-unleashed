from agents.nlp_agent import NLUModule


def test_parse_install():
    agent, payload = NLUModule.parse('please install dependencies')
    assert agent == 'installer'
    assert payload == 'please dependencies'


def test_parse_model_default():
    agent, payload = NLUModule.parse('which model should I use?')
    assert agent == 'model'
    assert 'which should i use?' == payload.lower()


from mhd.intent_spec import build_reply, intent_json_schema, parse_intent


def test_schema_strict():
    s = intent_json_schema()
    assert s["additionalProperties"] is False
    assert set(s["required"]) == {"action", "subject_phrase", "category", "color", "distance", "reason_code"}


def test_parse_and_reply():
    i = parse_intent('ok {"action":"follow","subject_phrase":"person in blue shirt","category":"person",'
                     '"color":"blue","distance":"medium","reason_code":"none"} done')
    assert i and build_reply(i) == "Following the person in blue shirt."
    assert parse_intent("not json") is None

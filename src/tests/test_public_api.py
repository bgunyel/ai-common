import ai_common


def test_public_api():
    expected_public_names = set(ai_common.__all__ + ['tools', 'utils', 'base'])
    public_attrs = {name for name in dir(ai_common) if not name.startswith('_')}

    unexpected = public_attrs - expected_public_names
    missing = expected_public_names - public_attrs

    assert not unexpected, f"❌ Unexpected public names found: {unexpected}"
    assert not missing, f"❌ Missing expected public names: {missing}"

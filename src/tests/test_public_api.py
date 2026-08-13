import subprocess
import sys

import pytest

import ai_common


def _probe(statement: str) -> set[str]:
    """Run `statement` in a fresh interpreter, return the modules it loaded.

    A subprocess is the only honest way to ask what an import pulls in: by
    the time pytest is running, the test process has already imported half
    the dependency tree for its own reasons.
    """
    source = f'import sys\n{statement}\nprint("\\n".join(sys.modules))\n'
    completed = subprocess.run(
        [sys.executable, '-c', source],
        capture_output=True, text=True, check=True,
    )
    return set(completed.stdout.split())


#: Loading any of these means the lazy `__init__` has stopped being lazy.
#: `langchain_core` is the expensive one and the reason this exists; the rest
#: are the other module-scope dependencies of the submodules `enums` does not
#: need. Each maps to the submodule that would have dragged it in.
_HEAVY_MODULES = {
    'langchain_core': 'base / llm / utils',
    'langchain_anthropic': 'llm',
    'langchain_openai': 'llm',
    'langchain_google_genai': 'llm',
    'langchain_groq': 'llm',
    'langchain_ollama': 'llm',
    'langchain_openrouter': 'llm',
    'ollama': 'llm / tools',
    'tqdm': 'tools',
    'tavily': 'utils / web_search',
    'PIL': 'utils',
}


def test_public_api():
    expected_public_names = set(ai_common.__all__)
    public_attrs = {name for name in dir(ai_common) if not name.startswith('_')}

    unexpected = public_attrs - expected_public_names
    missing = expected_public_names - public_attrs

    assert not unexpected, f"❌ Unexpected public names found: {unexpected}"
    assert not missing, f"❌ Missing expected public names: {missing}"


def test_all_and_the_lazy_module_map_agree():
    """The two lists in `__init__` are pinned to each other.

    A name in `__all__` but not the map is unreachable; a name in the map but
    not `__all__` is invisible to `dir()` and `import *`. Neither shows up as
    a failure anywhere else until a caller trips over it.
    """
    assert set(ai_common.__all__) == set(ai_common._MODULE_BY_NAME)


@pytest.mark.parametrize('name', ai_common.__all__)
def test_every_exported_name_resolves(name):
    """Every advertised name is actually importable.

    Guards the failure mode the lazy form introduces and the eager form could
    not have: a typo in `_MODULE_BY_NAME` is invisible until someone asks for
    that one name.
    """
    assert getattr(ai_common, name) is not None


def test_a_resolved_name_is_cached_in_the_package_namespace():
    """`__getattr__` binds what it resolves, so it runs once per name."""
    import ai_common as fresh
    fresh.__dict__.pop('LlmServers', None)

    assert 'LlmServers' not in vars(fresh)
    resolved = fresh.LlmServers
    assert vars(fresh).get('LlmServers') is resolved


def test_an_unknown_name_raises_attribute_error():
    """Not KeyError — `hasattr` and `getattr(..., default)` swallow only this."""
    with pytest.raises(AttributeError):
        ai_common.no_such_name

    assert not hasattr(ai_common, 'no_such_name')


def test_importing_the_package_loads_no_submodule():
    loaded = _probe('import ai_common')

    submodules = {m for m in loaded if m.startswith('ai_common.')}
    assert not submodules, f"❌ `import ai_common` eagerly loaded: {submodules}"


def test_reaching_for_a_cheap_name_loads_only_its_submodule():
    """The payoff: config modules want the enums and nothing else."""
    loaded = _probe('from ai_common import LlmServers, ModelNames')

    heavy = {m: owner for m, owner in _HEAVY_MODULES.items() if m in loaded}
    assert not heavy, f"❌ Importing the enums pulled: {heavy}"

    assert {m for m in loaded if m.startswith('ai_common.')} == {'ai_common.enums'}


def test_reaching_for_a_heavy_name_still_loads_its_submodule():
    """The other half of the contract: laziness defers work, it does not skip it."""
    loaded = _probe('from ai_common import get_llm')

    assert 'ai_common.llm' in loaded
    assert 'langchain_core' in loaded


def test_a_submodule_import_is_unaffected():
    """`from ai_common.llm import get_llm` is not routed through `__getattr__`."""
    loaded = _probe('from ai_common.llm import get_llm')

    assert 'ai_common.llm' in loaded
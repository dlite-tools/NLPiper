import builtins

import pytest


@pytest.fixture(scope="function")
def hide_available_pkg(request, monkeypatch):
    import_orig = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name in (request.param, ):
            raise ModuleNotFoundError()
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mocked_import)

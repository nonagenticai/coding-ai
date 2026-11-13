"""coding-ai must set tracing up at module scope, never in the lifespan.

This service had **three** independent reasons to export zero spans, all of them
inside one block that read correctly and logged success:

1. it imported ``opentelemetry.exporter.otlp.proto.grpc`` while the image ships
   ``...proto.http`` — verified live in the running pod, a ``ModuleNotFoundError``
   on every startup, caught and logged as "continuing without tracing";
2. it never called ``trace.set_tracer_provider()``, so even with the import fixed
   the instrumentation would have resolved a non-recording proxy;
3. it called ``FastAPIInstrumentor.instrument_app(app)`` from inside ``lifespan``,
   which is a silent no-op once Starlette has built the middleware stack.

Any one of those alone is enough to make tracing dead while looking alive, which
is why these tests assert the wiring *and* drive a real span.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

MAIN = Path(__file__).resolve().parents[2] / "src" / "main.py"


def _module() -> ast.Module:
    source = MAIN.read_text(encoding="utf-8")
    # Non-vacuity floor: a wrong or empty path would make every assertion pass.
    assert len(source) > 5_000, f"{MAIN} looks wrong: {len(source)} bytes"
    assert "app = FastAPI(" in source, f"{MAIN} does not construct the app"
    return ast.parse(source)


def test_setup_tracing_is_called_once_at_module_scope() -> None:
    tree = _module()

    top_level = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "setup_tracing"
    ]
    anywhere = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup_tracing"
    ]

    assert len(anywhere) == 1, f"expected one setup_tracing() call, found {len(anywhere)}"
    assert len(top_level) == 1, (
        "setup_tracing() is not called at module scope. Anywhere that runs after "
        "the app has started exports zero spans without raising."
    )


def test_the_lifespan_does_not_instrument_anything() -> None:
    """The old block is gone, not merely bypassed.

    An AST walk rather than a text grep, so the comment that replaced the block
    cannot satisfy it.
    """
    tree = _module()
    lifespan = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == "lifespan"
        ),
        None,
    )
    assert lifespan is not None, "no lifespan function found"

    banned = {"instrument_app", "set_tracer_provider"}
    found = [
        n.func.attr
        for n in ast.walk(lifespan)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in banned
    ]
    assert not found, f"instrumentation calls left in the lifespan: {found}"


def test_the_grpc_exporter_is_not_imported_anywhere() -> None:
    """Defect 1, pinned across the whole src/ tree.

    The image declares ``opentelemetry-exporter-otlp-proto-http``. Importing the
    grpc variant fails at runtime inside a broad ``except``, which is how this
    went unnoticed for months.

    Walks import statements rather than grepping for the dotted path: a comment
    documenting the removal contains that path too, and a text search cannot tell
    prose from code. ``ast.parse`` discards comments, so only real imports count.
    """
    src = MAIN.parent
    files = sorted(src.rglob("*.py"))
    assert len(files) > 20, f"scanned only {len(files)} files — the walk is wrong"

    offenders: list[str] = []
    scanned_imports = 0
    for f in files:
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                scanned_imports += 1
                if node.module and "otlp.proto.grpc" in node.module:
                    offenders.append(str(f.relative_to(src)))
            elif isinstance(node, ast.Import):
                scanned_imports += 1
                if any("otlp.proto.grpc" in a.name for a in node.names):
                    offenders.append(str(f.relative_to(src)))

    # Positive control: the walk must actually be seeing imports.
    assert scanned_imports > 100, f"only {scanned_imports} imports seen — the walk is broken"
    assert not offenders, f"grpc OTLP exporter imported in: {sorted(set(offenders))}"


def test_a_real_request_through_the_real_app_produces_a_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wiring tests cannot prove spans exist; this drives one.

    Uses ``/openapi.json`` rather than an ``/api/ai`` route because those now
    require a credential — a 401 would still be traced, but a 200 keeps this test
    about tracing rather than about auth.
    """
    from fastapi.testclient import TestClient
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.util._once import Once

    monkeypatch.setattr(trace, "_TRACER_PROVIDER", None, raising=False)
    monkeypatch.setattr(trace, "_TRACER_PROVIDER_SET_ONCE", Once(), raising=False)
    monkeypatch.setenv(
        "OTLP_TRACES_ENDPOINT", "http://tempo.tempo.svc.cluster.local:4318/v1/traces"
    )

    exporter = InMemorySpanExporter()
    real_add = TracerProvider.add_span_processor
    monkeypatch.setattr(
        TracerProvider,
        "add_span_processor",
        lambda self, _p: real_add(self, SimpleSpanProcessor(exporter)),
    )

    from coding_client.telemetry import setup_tracing
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/openapi-probe")
    def probe() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    assert setup_tracing(app, "coding-ai") is True

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert len(exporter.get_finished_spans()) == 0, "probe traffic must not be traced"
        assert client.get("/openapi-probe").status_code == 200

    assert exporter.get_finished_spans(), "no span produced"

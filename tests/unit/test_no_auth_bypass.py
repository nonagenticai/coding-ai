"""This service's vendored auth fork must not regrow the SKIP_AUTH hatch.

coding-client 1.16.0 deleted the hatch from the SDK and guards it there. That
guard **cannot see this file**: `src/auth/tenant.py` is a deliberate local fork
(it narrows `get_current_user_with_tenant` so an unverified `X-User-Roles: admin`
cannot escalate, and it stashes the resolved tenant in a contextvar the WM shim
forwards). The fork is justified — but it means this service carried its OWN
copy of

    if os.getenv("SKIP_AUTH", "").lower() == "true":
        return "test-tenant"

in `_resolve_tenant`, and deleting the SDK's three would have left this one live.
That is not hypothetical for this repo: #53 landed because this same fork had
silently missed the on-behalf-of rule the SDK already implemented.

So the fork gets its own guard, scoped to `src/auth/`, with the same three rules
as the SDK's and the same shape-first reasoning: a rename must not defeat it.

Mutants run against this file:

    M1  restore the branch verbatim                  A, B, C fire
    M2  same branch renamed CODING_TEST_MODE         B, C fire  (A silent)
    M3  renamed AND returning "fallback-tenant"      B fires    (A, C silent)
    M4  scan root pointed at an empty directory      non-vacuity fires
    M5  the existing trust_tenant_header/_is_trusted_network reads  stay GREEN
"""

from __future__ import annotations

import ast
import pathlib

AUTH_DIR = pathlib.Path(__file__).resolve().parents[2] / "src" / "auth"

DENY_NAMES = {
    "SKIP_AUTH",
    "BYPASS_AUTH",
    "AUTH_BYPASS",
    "DISABLE_AUTH",
    "AUTH_DISABLED",
    "NO_AUTH",
}

# A known env read that must be found — proof the AST walk parsed anything.
#
# MEASURED, not assumed. The whole of src/auth/ reads exactly two names:
# TRUSTED_CIDRS (tenant.py) and ENVIRONMENT (_default_tenant.py). The obvious
# choice, TRUST_TENANT_HEADER, is NOT read here — _header_trust.py delegates to
# coding_client — and picking it made this control fail on the first run. Which
# is the control working: an assertion that can never hold is worse than none,
# because it turns every future run red for the wrong reason.
CONTROL_ENV_NAME = "TRUSTED_CIDRS"

IDENTITY_CALLS = {"User"}


def _literal_str(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _env_read_names(node: ast.AST) -> list[str | None]:
    names: list[str | None] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            if isinstance(func, ast.Attribute) and func.attr in {"getenv", "get"}:
                target = func.value
                is_environ = (isinstance(target, ast.Attribute) and target.attr == "environ") or (
                    isinstance(target, ast.Name) and target.id == "os"
                )
                if is_environ and sub.args:
                    names.append(_literal_str(sub.args[0]))
        elif isinstance(sub, ast.Subscript):
            target = sub.value
            if isinstance(target, ast.Attribute) and target.attr == "environ":
                names.append(_literal_str(sub.slice))
    return names


def _returns_identity_constant(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.Return) or stmt.value is None:
        return False
    value = stmt.value
    if isinstance(value, ast.Constant):
        return isinstance(value.value, str) and bool(value.value)
    if isinstance(value, ast.Call):
        func = value.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        return name in IDENTITY_CALLS
    return False


def _parsed() -> list[tuple[pathlib.Path, ast.Module]]:
    return [
        (p, ast.parse(p.read_text(), filename=str(p)))
        for p in sorted(AUTH_DIR.rglob("*.py"))
        if "__pycache__" not in p.parts
    ]


def test_the_scan_is_not_vacuous():
    """Mutant M4 must land HERE, not sail through three green rules."""
    trees = _parsed()
    files = [p for p, _ in trees]
    assert AUTH_DIR.is_dir(), f"{AUTH_DIR} does not exist — the fork moved"
    assert len(files) >= 3, f"only {len(files)} file(s) under {AUTH_DIR}"
    assert AUTH_DIR / "tenant.py" in files, "the forked file is not in scope"

    found = {name for _, tree in trees for name in _env_read_names(tree)}
    assert CONTROL_ENV_NAME in found, (
        f"POSITIVE CONTROL FAILED: {CONTROL_ENV_NAME} not found in any env read. "
        "The AST walk is not seeing env reads, so the rules below cannot fire."
    )


def test_no_bypass_variable_is_read_in_the_fork():
    offenders = [
        f"{path.name} reads {name}"
        for path, tree in _parsed()
        for name in _env_read_names(tree)
        if name in DENY_NAMES
    ]
    assert not offenders, (
        "An auth-bypass environment variable is read in this service's auth fork:\n  "
        + "\n  ".join(offenders)
        + "\nUse coding_client.testing.override_tenant_auth in tests instead — it is "
        "per-app and cannot be switched on by the environment. When overriding this "
        "service's require_tenant, remember to set _request_tenant_id_var."
    )


def test_no_env_gated_identity_shortcut_in_the_fork():
    """Name-independent — this is what survives a rename (mutants M2/M3).

    An env-gated EARLY RETURN is fine; returning an IDENTITY is not.
    """
    offenders = []
    for path, tree in _parsed():
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            body = [s for s in node.body if not isinstance(s, ast.Expr | ast.Pass)]
            if not body or not isinstance(body[0], ast.If):
                continue
            first = body[0]
            if not _env_read_names(first.test):
                continue
            inner = [s for s in first.body if not isinstance(s, ast.Expr | ast.Pass)]
            if len(inner) == 1 and _returns_identity_constant(inner[0]):
                offenders.append(f"{path.name}:{first.lineno} in {node.name}()")
    assert not offenders, (
        "A function in the auth fork opens with an environment-gated shortcut that "
        "returns an identity:\n  "
        + "\n  ".join(offenders)
        + "\nThat is the SKIP_AUTH shape whatever the variable is called: one env var "
        "and every caller resolves to a constant tenant with no credential."
    )


def test_the_fork_does_not_carry_the_tenant_it_would_hand_out():
    offenders = [
        f"{path.name}:{node.lineno}"
        for path, tree in _parsed()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value == "test-tenant"
    ]
    assert not offenders, (
        'The literal "test-tenant" is present in the auth fork at:\n  '
        + "\n  ".join(offenders)
        + "\nA production auth path has no reason to carry the tenant it would hand out."
    )

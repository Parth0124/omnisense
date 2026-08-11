"""`/api/v1/projects`: status codes, scopes, and the shape of the wire contract.

The service tests next door cover the behaviour. These cover the parts only the
HTTP layer decides -- which status code a conflict becomes, which scope each verb
needs, and whether the response says what the schema promises. Those are the
things a client depends on and the service cannot enforce.
"""

from __future__ import annotations

import httpx
import pytest

from backend.api.deps import ROLE_SCOPES, SCOPES
from backend.api.v1.projects import get_project_service
from backend.core.exceptions import ConflictError, NotFoundError
from backend.main import create_app
from models.project import Project, ProjectSource

pytestmark = pytest.mark.unit


class FakeProjectService:
    """Enough of `ProjectService` to exercise the routes.

    A fake rather than a real service against SQLite, because what is under test
    here is the translation -- a `ConflictError` becoming 409, a `NotFoundError`
    becoming 404 -- and a real service would make those depend on database state
    two layers away from the assertion.
    """

    def __init__(self) -> None:
        self.projects: dict[str, Project] = {}
        self.attached: list[tuple[str, str]] = []
        self.detached: list[str] = []

    def _project(self, slug: str) -> Project:
        if slug not in self.projects:
            raise NotFoundError.for_resource("project", slug)
        return self.projects[slug]

    async def create(self, *, name: str, slug=None, description=None, metadata=None) -> Project:
        resolved = slug or name.lower().replace(" ", "-")
        if resolved in self.projects:
            raise ConflictError(f"a project with slug {resolved!r} already exists.")
        project = Project(
            id=f"prj_{resolved}",
            tenant_id="local",
            slug=resolved,
            name=name,
            description=description,
        )
        self.projects[resolved] = project
        return project

    async def list(self, *, include_inactive: bool = False) -> list[Project]:
        return [p for p in self.projects.values() if include_inactive or p.is_active]

    async def get(self, slug: str) -> Project:
        return self._project(slug)

    async def sources(self, slug: str) -> list[ProjectSource]:
        self._project(slug)
        return [
            ProjectSource(
                source_id="src_1",
                project_id=f"prj_{slug}",
                platform="github",
                name="omnisense/api",
                artifact_count=7,
            )
        ]

    async def attach_source(self, *, slug: str, source_id: str) -> ProjectSource:
        self._project(slug)
        self.attached.append((slug, source_id))
        return ProjectSource(
            source_id=source_id,
            project_id=f"prj_{slug}",
            platform="github",
            name="omnisense/api",
        )

    async def detach_source(self, *, source_id: str) -> None:
        self.detached.append(source_id)

    async def set_active(self, *, slug: str, is_active: bool) -> Project:
        project = self._project(slug)
        updated = project.model_copy(update={"is_active": is_active})
        self.projects[slug] = updated
        return updated


@pytest.fixture
def fake() -> FakeProjectService:
    return FakeProjectService()


@pytest.fixture
async def client(fake: FakeProjectService) -> httpx.AsyncClient:
    app = create_app()
    app.dependency_overrides[get_project_service] = lambda: fake
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def auth(role: str = "admin") -> dict[str, str]:
    import time

    from backend.core.config import get_settings
    from backend.core.security import encode_jws

    now = int(time.time())
    token = encode_jws(
        {"sub": "t", "tenant": "local", "role": role, "iat": now, "exp": now + 600},
        secret=get_settings().security.secret_key.get_secret_value(),
    )
    return {"Authorization": f"Bearer {token}"}


class TestScopes:
    """The vocabulary is closed and `require_scopes` raises at import on a typo,
    so the risk is not a bad name -- it is a verb wired to the wrong one."""

    def test_the_project_scopes_are_in_the_vocabulary(self) -> None:
        assert {"projects:read", "projects:write"} <= SCOPES

    def test_a_viewer_can_read_projects_but_not_change_them(self) -> None:
        """Hiding the project list from a viewer would leave them able to read
        artifacts and unable to say what those artifacts belong to."""
        assert "projects:read" in ROLE_SCOPES["viewer"]
        assert "projects:write" not in ROLE_SCOPES["viewer"]

    def test_an_analyst_can_onboard_repositories(self) -> None:
        """Ordinary use of the product, not an administrative act."""
        assert "projects:write" in ROLE_SCOPES["analyst"]

    async def test_writing_without_the_scope_is_403(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/projects", json={"name": "Nope"}, headers=auth("viewer")
        )
        assert response.status_code == 403

    async def test_reading_without_a_token_is_401(self, client: httpx.AsyncClient) -> None:
        assert (await client.get("/api/v1/projects")).status_code == 401


class TestCreate:
    async def test_creating_returns_201_and_the_derived_slug(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.post(
            "/api/v1/projects", json={"name": "OmniSense API"}, headers=auth()
        )
        assert response.status_code == 201
        assert response.json()["slug"] == "omnisense-api"

    async def test_a_duplicate_slug_is_409_not_a_silent_adoption(
        self, client: httpx.AsyncClient
    ) -> None:
        """Running `omnisense init` twice with the same name almost always means
        the second run was a mistake, and quietly attaching new repositories to
        an existing project is the harder mistake to notice."""
        await client.post("/api/v1/projects", json={"name": "OmniSense"}, headers=auth())
        again = await client.post("/api/v1/projects", json={"name": "OmniSense"}, headers=auth())
        assert again.status_code == 409

    async def test_an_empty_name_is_rejected_by_the_schema(self, client: httpx.AsyncClient) -> None:
        response = await client.post("/api/v1/projects", json={"name": ""}, headers=auth())
        assert response.status_code == 422


class TestRead:
    async def test_detail_includes_sources_and_a_total(self, client: httpx.AsyncClient) -> None:
        await client.post("/api/v1/projects", json={"name": "OmniSense"}, headers=auth())
        body = (await client.get("/api/v1/projects/omnisense", headers=auth())).json()

        assert body["slug"] == "omnisense"
        assert [s["name"] for s in body["sources"]] == ["omnisense/api"]
        assert body["artifact_count"] == 7

    async def test_an_unknown_project_is_404(self, client: httpx.AsyncClient) -> None:
        assert (await client.get("/api/v1/projects/ghost", headers=auth())).status_code == 404

    async def test_listing_hides_paused_projects_by_default(
        self, client: httpx.AsyncClient
    ) -> None:
        await client.post("/api/v1/projects", json={"name": "Live"}, headers=auth())
        await client.post("/api/v1/projects", json={"name": "Paused"}, headers=auth())
        await client.post("/api/v1/projects/paused/deactivate", headers=auth())

        visible = (await client.get("/api/v1/projects", headers=auth())).json()
        assert [p["slug"] for p in visible] == ["live"]

        everything = (
            await client.get("/api/v1/projects?include_inactive=true", headers=auth())
        ).json()
        assert {p["slug"] for p in everything} == {"live", "paused"}


class TestMembership:
    async def test_attaching_is_idempotent_which_is_why_it_is_a_put(
        self, client: httpx.AsyncClient, fake: FakeProjectService
    ) -> None:
        await client.post("/api/v1/projects", json={"name": "OmniSense"}, headers=auth())
        for _ in range(2):
            response = await client.put("/api/v1/projects/omnisense/sources/src_1", headers=auth())
            assert response.status_code == 200
        assert fake.attached == [("omnisense", "src_1")] * 2

    async def test_detaching_removes_the_membership_not_the_source(
        self, client: httpx.AsyncClient, fake: FakeProjectService
    ) -> None:
        """`DELETE` on the membership path, deliberately. Modelling it as deleting
        the source would make the destructive reading the obvious one."""
        await client.post("/api/v1/projects", json={"name": "OmniSense"}, headers=auth())
        response = await client.delete("/api/v1/projects/omnisense/sources/src_1", headers=auth())
        assert response.status_code == 204
        assert fake.detached == ["src_1"]

    async def test_attaching_to_a_missing_project_is_404(self, client: httpx.AsyncClient) -> None:
        response = await client.put("/api/v1/projects/ghost/sources/src_1", headers=auth())
        assert response.status_code == 404


class TestThereIsNoDelete:
    async def test_deleting_a_project_is_not_a_route(self, client: httpx.AsyncClient) -> None:
        """Deleting would either orphan the sources or cascade into their
        artifacts, and "we stopped working on this" is a different fact from
        "this never happened"."""
        await client.post("/api/v1/projects", json={"name": "OmniSense"}, headers=auth())
        response = await client.delete("/api/v1/projects/omnisense", headers=auth())
        assert response.status_code == 405

    async def test_deactivate_and_activate_are_the_whole_vocabulary(
        self, client: httpx.AsyncClient
    ) -> None:
        await client.post("/api/v1/projects", json={"name": "OmniSense"}, headers=auth())

        paused = await client.post("/api/v1/projects/omnisense/deactivate", headers=auth())
        assert paused.json()["is_active"] is False

        resumed = await client.post("/api/v1/projects/omnisense/activate", headers=auth())
        assert resumed.json()["is_active"] is True

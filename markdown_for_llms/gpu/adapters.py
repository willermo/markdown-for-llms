"""The adapters of the providers of the GPU presets (MFL051 B; the interface D11, App. A of the plan of MFL051).

One face for the four providers of `presets.py`, the rented ones and the GPUs of the user:

    provider        "runpod" | "modal" | "local" | "remote"
    costs           True: a session, the ledger and the caps are mandatory
    arms            True: arm and disarm change the state of the provider
    leaves_machine  True: a transit declared, and confirmed for every job
    requirements(settings, key_names, endpoint) -> [Requirement]   pure: no network, no file
    read() -> Reading            read only, no spend
    verify_pins(preset) -> None  before every arm; ProviderError (`endpoint_out_of_pins`) otherwise
    arm(deadline, levers, fault, preset) -> info
    disarm() -> None             idempotent
    zero() -> (bool, state)      true when nothing billable exists any more; with the state read
    balance() -> Amount | None   None: no source, not applicable, or the key does not read the balance
    url

RunPod and Modal are the objects of `providers.py` wrapped without changing their behaviour: `arm`, `disarm`, `zero`
and `url` are theirs, and every call of the product to them runs in a child (`opchild`), as before. `read()` of an
adapter that costs never calls the URL of the service (on RunPod an arm is enough to start a worker, and a request to
the URL of a deployed Modal app starts a container): only the management APIs, the REST v2 of RunPod and the lists of
the CLI of Modal 1.6.3. `Reading` = {armed, busy, running, workers: [{id, status}], health, read_at}: `running`
counts the workers or containers RUNNING (`providers.running_count`) and is for the panel, never for the zero; the
zero is `zero()`, as the session reads it. `health` is `ok`, `starting`, `unhealthy` or `unknown`: from the statuses
listed for a rented GPU, from `GET /v1/models` of the server for a GPU of the user.

`balance()` reads with the key the worker already has, never another: RunPod `GET /v2/billing/balance` of the REST v2
(the permission `billing:read`, `runpod.interfaces.0021` of the GPU corpus), whose `balanceAmount` it returns; with a
key that cannot read it (401, 403) or any other failure, None, and no request leaves with another key. Modal has no
balance (`modal.costs.0014`, gap G31 closed): None, without a call. `providers.RunPod.balance()` (GraphQL) stays as it
is for the sessions of the command line and the endpoint reports: the product never calls it.

The local GPU (`gpu` profile of compose) and the remote server (`make server-up`) cost nothing and arm nothing: a
job of their presets opens no session and writes no row of `gpu_sessions` or of the ledger. The worker converts
with the endpoint and the keys of `direct(...)`, as `mdf convert` does with an endpoint without a provider. Their
`read()` asks `GET /v1/models` of the server: the local GPU on the Unix socket of its front, the remote one over
HTTPS with its key and its `ca_file`.

Standard library only at import; `httpx` is imported by the readings of a GPU of the user.
"""
import time

from . import presets, providers

HEALTH = ("ok", "starting", "unhealthy", "unknown")
READ_TIMEOUT_S = 5.0                # a reading of a GPU of the user: at most this long
ENV_FILE, SERVERS_FILE = ".env", "state/config/servers.json"


def reading(armed, busy, running, workers, health, read_at=None):
    """A `Reading` of the contract, closed: the workers as {id, status} only."""
    if health not in HEALTH:
        health = "unknown"
    return {"armed": bool(armed), "busy": bool(busy), "running": int(running),
            "workers": [{"id": w.get("id"), "status": w.get("status")} for w in workers or []],
            "health": health, "read_at": round(time.time() if read_at is None else read_at, 3)}


def health_of(workers):
    """The health of a rented GPU from the statuses of its workers or containers, never from its service URL."""
    statuses = {str(w.get("status") or "").upper() for w in workers or []}
    if "RUNNING" in statuses or "READY" in statuses:
        return "ok"
    if "UNHEALTHY" in statuses:
        return "unhealthy"
    if statuses & {"INITIALIZING", "PENDING", "THROTTLED"}:
        return "starting"
    return "unknown"


def requirement(name, where, kind):
    return {"name": name, "where": where, "kind": kind}


class Adapter:
    """The face of the contract (module docstring); the traits come from `presets.TRAITS`."""

    provider = None

    def __init__(self):
        traits = presets.TRAITS[self.provider]
        self.costs, self.arms, self.leaves_machine = traits["costs"], traits["arms"], traits["leaves_machine"]

    def key_names(self, endpoint=None):
        """The names of the keys the adapter reads from the keys file: those of the endpoint, else of the provider."""
        names = list(getattr(endpoint, "key_env", None) or presets.KEY_ENV[self.provider])
        return names

    def requirements(self, settings, key_names, endpoint):
        """What is missing for the preset of this provider to run here, without a network or a file: the endpoint of
        servers.json, the keys absent from `key_names` (the names the keys file sets, never a value) and, for a rented
        GPU, the caps of 0 USD. Each item {name, where, kind}, as `routestate.missing_of`."""
        out = []
        if endpoint is None:
            out.append(requirement("the endpoint of the GPU preset", SERVERS_FILE, "servers.json"))
        present = set(key_names or ())
        out += [requirement(n, ENV_FILE, "key") for n in self.key_names(endpoint) if n not in present]
        if self.costs and settings is not None:
            from decimal import Decimal
            for setting in ("gpu_cap_usd_" + self.provider, "gpu_session_cap_usd"):
                if Decimal(getattr(settings, setting)) <= 0:
                    out.append(requirement("MDF_" + setting.upper(), ENV_FILE, "setting"))
        return out

    def verify_pins(self, preset):
        return None

    def arm(self, deadline=None, levers="", fault="", preset=None):
        return {}

    def disarm(self):
        return None

    def zero(self):
        return True, {}

    def balance(self):
        return None

    @property
    def url(self):
        return None


class RunPod(Adapter):
    """`providers.RunPod`, wrapped: the REST v2 with the key of the worker."""

    provider = "runpod"

    def __init__(self, prov):
        super().__init__()
        self.prov = prov

    def read(self):
        state = self.prov.state()
        workers = state.get("workers") or []
        return reading(state.get("armed"), state.get("busy"), providers.running_count(state), workers,
                       health_of(workers))

    def verify_pins(self, preset):
        failed = providers.invariant_failures(providers.pin_checks(self.prov.endpoint(refresh=True),
                                                                   *self.prov.pin_facts(), preset=preset))
        if failed:
            raise providers.ProviderError("RunPod: the endpoint is outside its pins (%s)" % ", ".join(failed),
                                          op="runpod.pins")

    def arm(self, deadline=None, levers="", fault="", preset=None):
        return self.prov.arm(fault, False, levers, deadline=deadline, preset=preset)

    def disarm(self):
        self.prov.disarm()

    def zero(self):
        return self.prov.zero()

    def balance(self):
        """`GET /v2/billing/balance` with the key of the worker: {"usd", "read_at"}, or None when the key does not
        read it (401, 403) or the answer has no amount; never a second request, never another key."""
        try:
            doc = self.prov.call("GET", "/billing/balance")
        except providers.ProviderError:
            return None
        except Exception:                                # a transport, a JSON that is not one: no source
            return None
        amount = doc.get("balanceAmount") if isinstance(doc, dict) else None
        if isinstance(amount, bool) or not isinstance(amount, (int, float, str)):
            return None
        from decimal import Decimal, InvalidOperation
        try:
            value = Decimal(str(amount))
        except InvalidOperation:
            return None
        if not value.is_finite():
            return None
        return {"usd": str(value), "read_at": round(time.time(), 3)}

    @property
    def url(self):
        return self.prov.url


class Modal(Adapter):
    """`providers.Modal`, wrapped: the CLI 1.6.3 of the image, with the token of `modal_auth`."""

    provider = "modal"

    def __init__(self, prov):
        super().__init__()
        self.prov = prov

    def key_names(self, endpoint=None):
        # the proxy token of the endpoint, and the token of the SDK the CLI reads through `modal_auth`
        return super().key_names(endpoint) + [n for n in providers.MODAL_TOKEN]

    def read(self):
        state = self.prov.state()
        workers = state.get("workers") or []
        return reading(state.get("armed"), state.get("busy"), providers.running_count(state), workers,
                       health_of(workers))

    def verify_pins(self, preset):
        if not self.prov.volume_ok():
            raise providers.ProviderError("Modal: the volume of the weights is not in the workspace", op="modal.volume")
        self.prov.proxy_token()

    def arm(self, deadline=None, levers="", fault="", preset=None):
        return self.prov.arm(fault, False, levers, deadline=deadline)

    def disarm(self):
        self.prov.disarm()

    def zero(self):
        return self.prov.zero()

    def balance(self):
        return None                                      # no balance at Modal (modal.costs.0014): nothing is called

    @property
    def url(self):
        return self.prov.url


class Server(Adapter):
    """A GPU of the user: an endpoint of servers.json with its keys; nothing armed, nothing spent. `transport`: a
    transport of httpx for the tests."""

    def __init__(self, endpoint, keys=None, transport=None):
        super().__init__()
        self.endpoint, self.keys, self.transport = endpoint, keys or {}, transport

    def _client(self):
        import httpx
        from ..engine import relay, servers
        ep = self.endpoint
        verify = relay.authorities(ep)
        if self.transport is not None:
            transport = self.transport
        elif ep.kind == "gpu":
            transport = httpx.HTTPTransport(uds=ep.socket, retries=0)
        else:
            transport = httpx.HTTPTransport(verify=verify, retries=0)
        base = "http://gpu" if ep.kind == "gpu" else servers.canonical_url(ep.url)
        return httpx.Client(base_url=base, transport=transport, timeout=READ_TIMEOUT_S, trust_env=False,
                            follow_redirects=False, verify=verify, headers=servers.headers(ep, self.keys))

    def read(self):
        """`GET /v1/models` of the server: ok with a 200 that lists the model of the endpoint, starting with 204 or
        503 (the front waits for vLLM), unhealthy otherwise, unknown when it cannot be reached."""
        import httpx
        try:
            with self._client() as client:
                r = client.get("/v1/models")
                status, body = r.status_code, r.content[:1 << 20]
        except httpx.HTTPError:
            return reading(False, False, 0, [], "unknown")
        except Exception:                                # a ca_file that cannot be read, a socket path refused
            return reading(False, False, 0, [], "unknown")
        if status == 200:
            import json
            try:
                listed = [m.get("id") for m in (json.loads(body).get("data") or []) if isinstance(m, dict)]
            except (ValueError, AttributeError):
                listed = []
            health = "ok" if self.endpoint.model in listed else "unhealthy"
        elif status in (204, 503):
            health = "starting"
        else:
            health = "unhealthy"
        return reading(False, False, 1 if health == "ok" else 0, [], health)


class Local(Server):
    """The local GPU of the `gpu` profile, on the Unix socket of its front (`gpu/local_front.py`)."""

    provider = "local"


class Remote(Server):
    """The remote server of `make server-up`, over TLS (`ca_file`) with its bearer key."""

    provider = "remote"


def of_provider(prov):
    """The adapter of a rented provider of `providers.make` (RunPod or Modal)."""
    if prov.provider == "runpod":
        return RunPod(prov)
    if prov.provider == "modal":
        return Modal(prov)
    raise providers.ProviderError("provider runpod or modal")


def of_endpoint(provider, endpoint, keys=None, transport=None):
    """The adapter of a GPU of the user from its endpoint and its keys."""
    if provider == "local":
        return Local(endpoint, keys, transport)
    if provider == "remote":
        return Remote(endpoint, keys, transport)
    raise providers.ProviderError("provider local or remote")


def direct(settings, name, env_file):
    """What the worker converts with on a preset that costs nothing (rilievo H, J.7.3 of the plan): (endpoint, keys),
    the endpoint of servers.json that names the preset and its keys by name from the keys file. No session, no row of
    `gpu_sessions` or of the ledger. `lifecycle.Refusal` with the closed code otherwise; a rented preset is refused
    (`route_option_unsupported`): it runs only in its session."""
    from ..engine import servers
    from ..engine.codes import Code, EngineError
    from ..services import gpu_availability
    from . import lifecycle
    avail = gpu_availability(name, settings)
    if not avail["available"]:
        raise lifecycle.Refusal(avail["code"])
    table = presets.load()
    if presets.costs(table[name]):
        raise lifecycle.Refusal(Code.ROUTE_OPTION_UNSUPPORTED.value, "a rented GPU runs in its session")
    ep = lifecycle.endpoint_of(settings.servers, name, presets.endpoint_table(table))
    try:
        keys = servers.read_keys(ep, env_file)
    except EngineError as exc:
        raise lifecycle.Refusal(exc.code.value, "the keys of the GPU preset in the env file of the worker") from None
    return ep, keys

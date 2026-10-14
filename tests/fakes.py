"""Tiny stand-ins for streamlit, folium and streamlit_folium.

They let the UI modules be imported and EXECUTED in tests (catching typos,
wrong attribute names, broken control flow) without installing the real
libraries.  They do NOT verify that the real libraries accept the calls;
that is what a manual `streamlit run dashboard/app.py` is for.
"""

from __future__ import annotations

import sys
import types
from datetime import date


class Ctx:
    """Works as ``with ctx:`` and also answers the few methods called on it."""

    def __init__(self, fake=None):
        self.fake = fake

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, **kw):
        if self.fake:
            self.fake.log("status.update", **kw)

    def markdown(self, text, **kw):
        if self.fake:
            self.fake.log("markdown", text)


class AttrDict(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.calls: list = []
        self.session_state = AttrDict()
        self.buttons: dict = {}        # label -> bool
        self.values: dict = {}         # label -> returned widget value
        self.sidebar = Ctx(self)
        self.reject_stretch = False    # simulate an old Streamlit without width="stretch"

    # -- recording ---------------------------------------------------------------
    def log(self, name, *args, **kw):
        self.calls.append((name, args, kw))

    def of(self, name):
        return [c for c in self.calls if c[0] == name]

    def texts(self, name):
        return [str(c[1][0]) for c in self.of(name) if c[1]]

    # -- simple output ---------------------------------------------------------------
    def _out(name):
        def method(self, *a, **k):
            self.log(name, *a, **k)
        return method

    for _n in ("set_page_config", "title", "caption", "warning", "info", "error", "success", "markdown",
               "subheader", "header", "divider", "code", "metric", "write", "map"):
        locals()[_n] = _out(_n)
    del _n

    def image(self, *a, **k):
        if self.reject_stretch and k.get("width") == "stretch":
            raise TypeError("width must be int")
        self.log("image", *a, **k)

    def dataframe(self, *a, **k):
        if self.reject_stretch and k.get("width") == "stretch":
            raise TypeError("width must be int")
        self.log("dataframe", *a, **k)

    def download_button(self, label, *a, **k):
        self.log("download_button", label, **k)
        return False

    # -- inputs ----------------------------------------------------------------------------
    def text_input(self, label, value="", **k):
        return self.values.get(label, value)

    def number_input(self, label, value=0.0, **k):
        return self.values.get(label, value)

    def date_input(self, label, value=None, **k):
        return self.values.get(label, value)

    def selectbox(self, label, options, **k):
        return self.values.get(label, options[0])

    def multiselect(self, label, options, default=None, **k):
        return self.values.get(label, default or [])

    def radio(self, label, options, **k):
        return self.values.get(label, options[0])

    def slider(self, label, min_value=0, max_value=100, value=None, *a, **k):
        return self.values.get(label, value if value is not None else min_value)

    def checkbox(self, label, value=False, **k):
        return self.values.get(label, value)

    def button(self, label, **k):
        self.log("button", label, **k)
        return self.buttons.get(label, False)

    # -- containers ----------------------------------------------------------------------------
    def expander(self, *a, **k):
        return Ctx(self)

    def tabs(self, labels):
        self.log("tabs", labels)
        return [Ctx(self) for _ in labels]

    def columns(self, spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [Ctx(self) for _ in range(n)]

    def status(self, label, **k):
        self.log("status", label)
        return Ctx(self)

    def empty(self):
        return Ctx(self)

    def cache_data(self, *a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda fn: fn


class FakeFolium(types.ModuleType):
    def __init__(self):
        super().__init__("folium")
        self.calls: list = []
        fake = self

        def make(name):
            class Obj:
                def __init__(self, *args, **kwargs):
                    self.name, self.args, self.kwargs = name, args, kwargs
                    fake.calls.append((name, args, kwargs))

                def add_to(self, parent):
                    return self

                def add_child(self, child):
                    fake.calls.append(("add_child", (child,), {}))
                    return self

                def fit_bounds(self, bounds):
                    fake.calls.append(("fit_bounds", (bounds,), {}))

                def get_root(self):
                    return types.SimpleNamespace(html=self)
            Obj.__name__ = name
            return Obj

        for n in ("Map", "TileLayer", "FeatureGroup", "GeoJson", "GeoJsonPopup", "CircleMarker", "Marker",
                  "Icon", "Popup", "LayerControl", "Element"):
            setattr(self, n, make(n))
        self.raster_layers = types.SimpleNamespace(ImageOverlay=make("ImageOverlay"))

    def count(self, name):
        return len([c for c in self.calls if c[0] == name])


class FakeStreamlitFolium(types.ModuleType):
    def __init__(self, reject_stretch=False):
        super().__init__("streamlit_folium")
        self.calls: list = []
        self.reject_stretch = reject_stretch

    def st_folium(self, fmap, **kw):
        if self.reject_stretch and kw.get("width") == "stretch":
            raise TypeError("width must be int")
        self.calls.append(kw)
        return {}


def install():
    """Put the fakes into ``sys.modules``; returns ``(st, folium, st_folium_module)``."""
    st, fo, sf = FakeStreamlit(), FakeFolium(), FakeStreamlitFolium()
    sys.modules.update({"streamlit": st, "folium": fo, "streamlit_folium": sf})
    return st, fo, sf


def uninstall():
    """Remove the fakes AND forget the UI modules that were bound to them.

    Popping ``sys.modules`` is not enough: ``from dashboard import components``
    first looks for an attribute on the already-imported ``dashboard`` package,
    which would hand back the OLD module (still bound to the previous fake).
    """
    ui_modules = ("app", "components", "map_layers")
    for name in ("streamlit", "folium", "streamlit_folium", *(f"dashboard.{m}" for m in ui_modules)):
        sys.modules.pop(name, None)
    package = sys.modules.get("dashboard")
    for name in ui_modules:
        if package is not None and hasattr(package, name):
            delattr(package, name)

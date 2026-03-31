from project.__version__ import __version__


def version(request):
    return {"PRISMADB_VERSION": __version__}

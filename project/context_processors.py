from project.__version__ import __version__


def version(request):
    return {"PRISMA_VERSION": __version__}

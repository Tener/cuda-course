
def get():
    surf_def = filter( lambda s: s.startswith('SURF_'), open('surf.h').read().replace(',',' ').split() )
    return surf_def

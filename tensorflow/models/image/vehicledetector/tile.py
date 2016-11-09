def tile(start_x, start_y, end_x, end_y, tile_width, tile_height):
    for x in xrange(start_x, end_x, tile_width):
        for y in xrange(start_y, end_y, tile_height):
            yield (x,y)
    

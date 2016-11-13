def startposition(left_x, left_y, right_x, right_y, tile_width, tile_height, stride):
    for x in xrange(0, tile_width, stride):
        for y in xrange(0, tile_height, stride):
            if x+left_x+tile_width > right_x or y+left_y+tile_height > right_y:
                continue
            yield (left_x+x,left_y+y)

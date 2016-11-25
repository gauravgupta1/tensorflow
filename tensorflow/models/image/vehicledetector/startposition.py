def startposition(left_x, left_y, right_x, right_y, tile_width, tile_height, stride):
    for delta_x in xrange(0, tile_width, stride):
        for delta_y in xrange(0, tile_height, stride):
            if (delta_x + left_x + tile_width) > right_x or (delta_y + left_y + tile_height) > right_y:
                continue
            yield (left_x + delta_x,left_y + delta_y)

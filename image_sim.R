# simulate images
library(ggplot2)
library(dplyr)
library(raster)

grid_sim = function(xdim, ydim, offset_cap, square_size, train_lower, train_upper) {

    grid = expand.grid(
        x = seq(0, xdim),
        y = seq(0, ydim)
    )

    offset_x = sample(x = seq(0, offset_cap), size = 1)
    offset_y = sample(x = seq(0, offset_cap), size = 1)
    grid$z = 1
    grid$z[(grid$x + offset_x) %% square_size == 0 | (grid$y + offset_y) %% square_size == 0] = 0
    grid$train = (grid$x < train_lower | grid$x > train_upper) | (grid$y < train_lower | grid$y > train_upper)
    grid
}

grid = grid_sim(xdim = 100, ydim = 100, offset_cap = 2, square_size = 3, train_lower = 45, train_upper = 60)

grid$z = 255 * grid$z


ggplot(grid) +
    geom_tile(aes(x = x, y = y, fill = z, color = train)) +
    coord_equal()

train = grid %>% filter(train)
test = grid %>% filter(!train)

rast = raster::rasterFromXYZ(grid[, c('x', 'y', 'z')])
raster::writeRaster(rast, filename = 'test.png', format = "PNG", overwrite = TRUE)

dir.create('simimg', showWarnings = FALSE)

set.seed(123)

for (i in 1:100) {
    grid = grid_sim(
        xdim = 100,
        ydim = 100,
        offset_cap = sample(x = seq(0, 2), size = 1),
        square_size = sample(x = seq(2,5), size = 1),
        train_lower = 40,
        train_upper = 60
    )
    grid$z = 255 * grid$z

    rast = raster::rasterFromXYZ(grid[, c('x', 'y', 'z')])
    raster::writeRaster(rast, filename = file.path('simimg', sprintf('test%s.PNG', i)), format = "PNG", overwrite = TRUE)

    mask = grid[, c('x', 'y', 'train')]
    mask$train = as.integer(mask$train) * 255
    mask = raster::rasterFromXYZ(mask)
    raster::writeRaster(mask, filename = file.path('simimg', sprintf('mask%s.png', i)), format = "PNG", overwrite = TRUE)
}


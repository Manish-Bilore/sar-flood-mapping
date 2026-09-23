# helpers.R — shared plotting utilities for the event report (terra base maps + ggplot charts)
suppressPackageStartupMessages({
  library(terra); library(ggplot2); library(dplyr); library(tidyr); library(readr)
  library(jsonlite); library(patchwork); library(scales); library(knitr)
})

`%||%` <- function(a, b) if (is.null(a)) b else a
ROOT <- normalizePath(file.path(getwd(), ".."))          # report/ lives in the project root

ev_paths <- function(ev) {
  out <- file.path(ROOT, "outputs", ev)
  list(out = out, disp = file.path(out, "display"), tab = file.path(out, "tables"), zoom = file.path(out, "zoom"),
       fig = file.path(ROOT, "report", "figures"))
}

dsp <- function(P, n) { f <- file.path(P$disp, paste0(n, ".tif")); if (file.exists(f)) rast(f) else NULL }
tb  <- function(P, n) {
  f <- file.path(P$tab, paste0(n, ".csv"))
  if (!file.exists(f) || file.size(f) < 5) return(NULL)
  x <- read_csv(f, show_col_types = FALSE); if (nrow(x)) x else NULL
}
zm  <- function(P, z, n) { f <- file.path(P$zoom, z, paste0(n, ".tif")); if (file.exists(f)) rast(f) else NULL }
na_note <- function(what, how) cat(sprintf("\n> **%s not available yet** — run `%s`.\n\n", what, how))

PAL <- list(
  db    = gray.colors(100, start = 0, end = 1),
  flood = colorRampPalette(c("#f7f7f7", "#9ecae1", "#2171b5", "#08306b"))(100),
  div   = colorRampPalette(c("#b2182b", "#ef8a62", "#fddbc7", "#f7f7f7", "#d1e5f0", "#67a9cf", "#2166ac"))(100),
  coh   = hcl.colors(100, "Inferno"),
  hand  = hcl.colors(100, "YlGnBu", rev = TRUE),
  prob  = hcl.colors(100, "Blues 3", rev = TRUE)
)
WC <- data.frame(value = c(10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100),
                 class = c("tree", "shrub", "grass", "cropland", "built-up", "bare", "snow", "water", "herb. wetland", "mangrove", "moss"),
                 col = c("#006400", "#ffbb22", "#ffff4c", "#f096ff", "#fa0000", "#b4b4b4", "#f0f0f0", "#0064c8", "#0096a0", "#00cf75", "#fae6a0"))
METHOD_LAB <- c(otsu_vv = "Otsu VV", otsu_vh = "Otsu VH", split_vv = "Split-based VV", cd_vv = "Change detection VV",
                gee2024 = "2024 GEE logic (replica)", urban_incr = "Urban brightening (candidate)", final_majority = "Final (majority)")
mlab <- function(x) ifelse(x %in% names(METHOD_LAB), METHOD_LAB[x], gsub("^dl_", "DL: ", x))

# display label for a zoom window: config zoom_labels, else Title Case of the key
zlab <- function(S, z) {
  lab <- tryCatch(S$zoom_labels[[z]], error = function(e) NULL)
  if (!is.null(lab) && nzchar(lab)) return(lab)
  paste(toupper(substring(strsplit(z, "_")[[1]], 1, 1)), substring(strsplit(z, "_")[[1]], 2), sep = "", collapse = " ")
}

districts <- function(P, crs_to) {
  f <- file.path(P$out, "districts.geojson")
  if (file.exists(f)) project(vect(f), crs_to) else NULL
}

# continuous raster plot: values clamped into the colour range (terra's fill_range is version-dependent)
cplot <- function(r, col, range, title = "", leg = "", legend = TRUE, add = FALSE) {
  plot(clamp(r, range[1], range[2], values = TRUE), col = col, range = range, type = "continuous", main = title,
       axes = FALSE, box = FALSE, cex.main = 1.1, legend = legend, add = add, plg = list(title = leg, cex = 0.85))
}

# grid of maps sharing one colour scale
mapgrid <- function(rs, titles, col, range, ncol = 3, leg = "", dist = NULL, mar = c(1, 1, 2.2, 4.5)) {
  keep <- !vapply(rs, is.null, TRUE); rs <- rs[keep]; titles <- titles[keep]
  if (!length(rs)) return(invisible())
  op <- par(mfrow = c(ceiling(length(rs) / ncol), min(ncol, length(rs))), mar = mar); on.exit(par(op))
  for (i in seq_along(rs)) {
    cplot(rs[[i]], col, range, titles[i], leg)
    if (!is.null(dist)) lines(dist, col = "#222222", lwd = 0.6)
  }
}

wc_plot <- function(r, title = "ESA WorldCover 2021", dist = NULL) {
  v <- WC[WC$value %in% unique(values(r, na.rm = TRUE)), ]
  levels(r) <- data.frame(value = v$value, class = v$class)
  coltab(r) <- data.frame(value = v$value, col = v$col)
  plot(r, main = title, axes = FALSE, box = FALSE, plg = list(cex = 0.75))
  if (!is.null(dist)) lines(dist, col = "#222222", lwd = 0.6)
}

rgb_change <- function(dry_db, co_db, lo = -25, hi = 0, title = "") {
  s <- function(r) clamp((r - lo) / (hi - lo) * 255, 0, 255)
  plotRGB(c(s(dry_db), s(co_db), s(co_db)), r = 1, g = 2, b = 3, scale = 255, axes = FALSE, main = title, mar = c(1, 1, 2.2, 1))
}

theme_rep <- function() theme_minimal(base_size = 10) + theme(panel.grid.minor = element_blank(), legend.position = "bottom",
                                                             plot.title = element_text(face = "bold", size = 11))

kbl <- function(df, digits = 2, ...) knitr::kable(df, digits = digits, format.args = list(big.mark = ","), ...)

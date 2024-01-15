
# Step 1: Download data ----

download_season <- function (year) {

  schedule_filter <- glue::glue("sportId=1&gameType=R&startDate=01/01/{year}&endDate=12/31/{year}")
  endpoint <- glue::glue("http://statsapi.mlb.com:80/api/v1/schedule?{schedule_filter}")
  schedule_json <- jsonlite::fromJSON(endpoint, flatten = TRUE)

  schedule <- do.call(dplyr::bind_rows, args = schedule_json$dates$games)

  # If no game has a `resumeDate`, then this column will be NULL in the schedule dataframe.
  # Below, we filter out any non-NA values of `resumeDate`, so this column needs to be present.
  if (is.null(schedule$resumeDate)) {
    schedule$resumeDate <- NA
  }

  game <- schedule |>
    # Filter out non-NA resumeDate to get down to one row per game ID
    dplyr::filter(status.detailedState %in% c("Final", "Completed Early"), is.na(resumeDate)) |>
    dplyr::arrange(officialDate) |>
    dplyr::select(
      game_id = gamePk,
      year = season,
      date = officialDate,
      team_id_away = teams.away.team.id,
      team_name_away = teams.away.team.name,
      team_id_home = teams.home.team.id,
      team_name_home = teams.home.team.name,
      venue_id = venue.id,
      score_away = teams.away.score,
      score_home = teams.home.score
    )

  return(game)
}

game <- NULL

for (year in seq(2023, 2021)) {
  print(year)
  game <- rbind(game, download_season(year))
}


# Step 2: Calculate record ----

calculate_record <- function(game, alpha) {

  data_away <- game |>
    dplyr::group_by(year, team_id = team_id_away, team_name = team_name_away) |>
    dplyr::summarize(
      games = dplyr::n(),
      wins = sum(score_away > score_home),
      runs_scored = sum(score_away),
      runs_allowed = sum(score_home),
      .groups = "drop"
    )
  
  data_home <- game |>
    dplyr::group_by(year, team_id = team_id_home, team_name = team_name_home) |>
    dplyr::summarize(
      games = dplyr::n(),
      wins = sum(score_home > score_away),
      runs_scored = sum(score_home),
      runs_allowed = sum(score_away),
      .groups = "drop"
    )
  
  data <- dplyr::bind_rows(data_away, data_home) |>
    dplyr::group_by(year, team_id, team_name) |>
    dplyr::summarize(
      games = sum(games),
      wins = sum(wins),
      runs_scored = sum(runs_scored),
      runs_allowed = sum(runs_allowed),
      .groups = "drop"
    ) |>
    dplyr::mutate(
      win_pct_actual = wins / games,
      win_pct_pythag = runs_scored^alpha / (runs_scored^alpha + runs_allowed^alpha),
      win_pct_residual = win_pct_actual - win_pct_pythag
    )
  
  return(data)
}

data <- calculate_record(game, alpha = 2)

with(data, plot(win_pct_pythag, win_pct_actual))


# Step 3: Calculate optimal Pythagorean exponent ----

alpha_seq <- seq(1.5, 2.0, by = 0.01)
error <- rep(NA, length(alpha_seq))

for (i in 1:length(alpha_seq)) {

  error[i] <- game |>
    calculate_record(alpha = alpha_seq[i]) |>
    dplyr::summarize(error = sqrt(mean((win_pct_actual - win_pct_pythag)^2))) |>
    dplyr::pull(error)
}

plot(alpha_seq, error)

alpha_seq[which.min(error)]

alpha <- 1.72


# Step 4: Calculate noise variance of residual winning pct ----

bootstrap_data <- NULL
B <- 100

for (b in 1:B) {

  bootstrap_sample <- game |>
    dplyr::slice(sample(1:nrow(game), size = nrow(game), replace = TRUE)) |>
    calculate_record(alpha = 1.72) |>
    dplyr::mutate(
      win_pct_actual = wins / games,
      win_pct_pythag = runs_scored^alpha / (runs_scored^alpha + runs_allowed^alpha),
      win_pct_residual = win_pct_actual - win_pct_pythag
    ) |>
    tibble::add_column(bootstrap_sample = b, .before = 1)

  bootstrap_data <- dplyr::bind_rows(bootstrap_data, bootstrap_sample)
}

sd_residual_noise <- bootstrap_data |>
  dplyr::group_by(year, team_id, team_name) |>
  dplyr::summarize(sd_residual_noise = sd(win_pct_residual), .groups = "drop") |>
  dplyr::summarize(sd_residual_noise = mean(sd_residual_noise), .groups = "drop") |>
  dplyr::pull(sd_residual_noise)


# Step 5: Calculate signal variance of residual winning pct ----

sd_residual_signal <- sqrt(mean(data$win_pct_residual^2) - sd_residual_noise^2)


# Step 6: Calculate sample size at which we prefer actual winning pct ----

n <- sd_residual_noise^2 * 162 / sd_residual_signal^2

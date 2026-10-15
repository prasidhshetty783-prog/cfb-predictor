---
tags: [data, audit]
updated: 2026-10-15T00:12:19+00:00
status: PASS
---
# Data Quality Report

> [!summary] In plain words
> Before any model trains, we check the data like a referee checks equipment.
> Overall result: **PASS**. Hard problems found: none.

## Hard checks
- Duplicate games: **0**
- Tied FBS games (impossible since overtime began in 1996): **0** (small-college ties ignored: 2)
- Team playing itself: **0**
- Playoff games per season: {2014: 3, 2015: 3, 2016: 3, 2017: 3, 2018: 3, 2019: 3, 2020: 3, 2021: 3, 2022: 3, 2023: 3, 2024: 11, 2025: 11}
- Leak-only columns present in the games table (allowed there, banned as features): away_points, home_points, home_win, margin

## Champions in our data vs real history
| Season | Real champion | Our data | Match |
|---|---|---|---|
| 2004 | USC | USC | yes |
| 2005 | Texas | Texas | yes |
| 2006 | Florida | Florida | yes |
| 2007 | LSU | LSU | yes |
| 2008 | Florida | Florida | yes |
| 2009 | Alabama | Alabama | yes |
| 2010 | Auburn | Auburn | yes |
| 2011 | Alabama | Alabama | yes |
| 2012 | Alabama | Alabama | yes |
| 2013 | Florida State | Florida State | yes |
| 2014 | Ohio State | Ohio State | yes |
| 2015 | Alabama | Alabama | yes |
| 2016 | Clemson | Clemson | yes |
| 2017 | Alabama | Alabama | yes |
| 2018 | Clemson | Clemson | yes |
| 2019 | LSU | LSU | yes |
| 2020 | Alabama | Alabama | yes |
| 2021 | Georgia | Georgia | yes |
| 2022 | Georgia | Georgia | yes |
| 2023 | Michigan | Michigan | yes |
| 2024 | Ohio State | Ohio State | yes |
| 2025 | Indiana | Indiana | yes |

## Coverage per season (completed games with at least one FBS team)
Columns ending in `_cov` = share of FBS teams that have that data. `market_line` = share of
games with a betting line (benchmark only). `adv_stats_both` = both teams have advanced stats.

|   season |   games |   fbs_vs_fbs |   postseason |   home_win_rate |   market_line |   adv_stats_both |   talent_cov |   ret_all_cov |   portal_net_cov |   prev_adj_net_cov |
|---------:|--------:|-------------:|-------------:|----------------:|--------------:|-----------------:|-------------:|--------------:|-----------------:|-------------------:|
|     2001 |     684 |          627 |            0 |           0.614 |         0     |            0     |        0     |         0     |                0 |              0     |
|     2002 |     744 |          679 |            0 |           0.633 |         0     |            0     |        0     |         0     |                0 |              0     |
|     2003 |     743 |          671 |            0 |           0.631 |         0     |            0     |        0     |         0     |                0 |              0     |
|     2004 |     721 |          655 |           28 |           0.652 |         0     |            0.981 |        0     |         0.915 |                0 |              0     |
|     2005 |     718 |          664 |           28 |           0.596 |         0     |            1     |        0.992 |         1     |                0 |              0.983 |
|     2006 |     792 |          714 |           32 |           0.624 |         0.475 |            0.995 |        0.992 |         1     |                0 |              1     |
|     2007 |     798 |          718 |           32 |           0.623 |         0.738 |            1     |        0.992 |         1     |                0 |              0.992 |
|     2008 |     805 |          717 |           34 |           0.629 |         0.393 |            1     |        0.992 |         1     |                0 |              0.992 |
|     2009 |     808 |          714 |           34 |           0.636 |         0.818 |            1     |        0.992 |         1     |                0 |              1     |
|     2010 |     808 |          718 |           35 |           0.605 |         0.834 |            1     |        0.992 |         1     |                0 |              1     |
|     2011 |     812 |          715 |           35 |           0.638 |         0.941 |            1     |        0.992 |         1     |                0 |              1     |
|     2012 |     840 |          732 |           35 |           0.613 |         0.95  |            1     |        0.992 |         0.992 |                0 |              0.968 |
|     2013 |     850 |          739 |           37 |           0.619 |         0.965 |            1     |        0.992 |         0.992 |                0 |              0.992 |
|     2014 |     868 |          760 |           39 |           0.606 |         0.97  |            0.999 |        0.992 |         1     |                0 |              0.977 |
|     2015 |     870 |          765 |           41 |           0.611 |         0.968 |            1     |        0.992 |         0.992 |                1 |              0.992 |
|     2016 |     873 |          760 |           41 |           0.625 |         0.968 |            1     |        0.992 |         1     |                1 |              1     |
|     2017 |     874 |          776 |           40 |           0.606 |         0.966 |            1     |        0.992 |         0.985 |                1 |              0.985 |
|     2018 |     884 |          772 |           39 |           0.635 |         0.977 |            1     |        0.992 |         1     |                1 |              0.992 |
|     2019 |     888 |          774 |           40 |           0.642 |         0.971 |            1     |        0.992 |         1     |                1 |              1     |
|     2020 |     568 |          534 |           26 |           0.585 |         1     |            1     |        0.992 |         1     |                1 |              1     |
|     2021 |     887 |          770 |           38 |           0.617 |         1     |            1     |        0.992 |         0.985 |                1 |              0.977 |
|     2022 |     896 |          776 |           42 |           0.625 |         1     |            1     |        0.992 |         0.992 |                1 |              0.992 |
|     2023 |     910 |          792 |           42 |           0.63  |         1     |            1     |        0.992 |         1     |                1 |              0.985 |
|     2024 |     919 |          798 |           46 |           0.632 |         1     |            1     |        0.993 |         1     |                1 |              0.993 |
|     2025 |     934 |          808 |           46 |           0.646 |         1     |            1     |        0.993 |         1     |                1 |              0.985 |
|     2026 |     448 |          329 |            0 |           0.717 |         1     |            1     |        0.993 |         1     |                1 |              0.986 |

## Live season (2026)
- Completed games: 448 (through week 6, last game 2026-10-11 02:30:00+00:00)
- Games still to play: 440

## Known gaps (and what we did about them)
- Bowl/playoff games for 2001-2003 are not in any source (Elo warm-up only).
- ESPN betting spreads are sign-flipped for ~1/3 of 2006-2012 games: only used 2013+.
- ESPN FPI weekly snapshots are often captured after the season: benchmark only.
- Talent composite is derived by SportsDataverse from recruiting data (2005+).
- Transfer portal data starts 2015 (meaningful from 2018).

Related: [[Data Sources]] · [[Feature Catalog]] · [[Evaluation Protocol]]

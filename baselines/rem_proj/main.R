## ================== SETUP ==================

suppressPackageStartupMessages({
  library(goldfish)
  library(readr)
  library(dplyr)
  library(tidyr)
  library(pROC)
})

K_PCA   <- 4
SEEDS   <- c(123, 456, 789)   # three seeds
K_NEG   <- 20

## ================== LOAD ==================

events_raw <- read_csv("../.cache/edges_partners.csv", show_col_types = FALSE)
nodes_raw  <- read_csv("../.cache/nodes_partners.csv", show_col_types = FALSE)

events <- events_raw %>%
  mutate(
    from     = as.character(source),
    to       = as.character(target),
    time_num = as.numeric(time),
    type     = as.character(type)
  ) %>%
  arrange(time_num)

edge_types <- sort(unique(events$type))
cat("Edge types:", paste(edge_types, collapse = ", "), "\n")
cat("Total events:", nrow(events), "\n")

node_type_lookup <- setNames(as.character(nodes_raw$type),
                             as.character(nodes_raw$node))
## Fill missing types
node_type_lookup[is.na(node_type_lookup) | node_type_lookup == ""] <- "unk"

pca_scores <- as.matrix(nodes_raw[, paste0("pc", seq_len(K_PCA))])
rownames(pca_scores) <- as.character(nodes_raw$node)
pca_scores[is.na(pca_scores)] <- 0

## ================== SHARED HELPERS ==================

build_type_objects <- function(etype, events_train, actors) {
  ev <- events_train %>%
    filter(type == etype) %>%
    transmute(time      = as.numeric(time_num),
              sender    = as.character(from),
              receiver  = as.character(to),
              increment = 1L) %>%
    as.data.frame(stringsAsFactors = FALSE)
  
  ev_name  <- paste0("ev_",  etype)
  net_name <- paste0("net_", etype)
  dep_name <- paste0("dep_", etype)
  
  assign(ev_name, ev, envir = globalenv())
  net <- defineNetwork(nodes = actors, directed = TRUE)
  assign(net_name, net, envir = globalenv())
  net <- eval(bquote(
    linkEvents(x = .(as.name(net_name)),
               changeEvents = .(as.name(ev_name)),
               nodes = actors)
  ), envir = globalenv())
  assign(net_name, net, envir = globalenv())
  dep <- eval(bquote(
    defineDependentEvents(events = .(as.name(ev_name)),
                          nodes = actors,
                          defaultNetwork = .(as.name(net_name)))
  ), envir = globalenv())
  assign(dep_name, dep, envir = globalenv())
  
  list(events = ev, etype = etype)
}

try_estimate <- function(formula_str, sub) {
  tryCatch(
    estimate(as.formula(formula_str),
             model = "DyNAM", subModel = sub,
             estimationInit = list(engine = "gather_compute")),
    error = function(e) NULL
  )
}

## ---- Formula building ----
## sim over PCA, same + ego + alter over node_type
sim_pc <- paste(sprintf("sim(actors$pc%d)", seq_len(K_PCA)), collapse = " + ")
ego_pc <- paste(sprintf("ego(actors$pc%d)", seq_len(K_PCA)), collapse = " + ")

fit_choice <- function(etype) {
  nn <- paste0("net_", etype); dn <- paste0("dep_", etype)
  specs <- list(
    ## Full: endogenous + PCA sim + same + ego/alter type
    sprintf("%s ~ inertia(%s) + recip(%s) + trans(%s) + %s + same(actors$node_type) + ego(actors$node_type) + alter(actors$node_type)",
            dn, nn, nn, nn, sim_pc),
    ## Drop alter
    sprintf("%s ~ inertia(%s) + recip(%s) + trans(%s) + %s + same(actors$node_type) + ego(actors$node_type)",
            dn, nn, nn, nn, sim_pc),
    ## Drop ego/alter, keep same
    sprintf("%s ~ inertia(%s) + recip(%s) + trans(%s) + %s + same(actors$node_type)",
            dn, nn, nn, nn, sim_pc),
    ## Drop trans
    sprintf("%s ~ inertia(%s) + recip(%s) + %s + same(actors$node_type)",
            dn, nn, nn, sim_pc),
    ## Drop recip
    sprintf("%s ~ inertia(%s) + %s + same(actors$node_type)",
            dn, nn, sim_pc),
    ## Endogenous + same only
    sprintf("%s ~ inertia(%s) + recip(%s) + same(actors$node_type)",
            dn, nn, nn),
    ## Inertia only (last resort)
    sprintf("%s ~ inertia(%s)", dn, nn)
  )
  for (i in seq_along(specs)) {
    res <- try_estimate(specs[[i]], "choice")
    if (!is.null(res)) { attr(res, "spec_used") <- specs[[i]]; return(res) }
  }
  NULL
}

fit_rate <- function(etype) {
  nn <- paste0("net_", etype); dn <- paste0("dep_", etype)
  specs <- list(
    sprintf("%s ~ indeg(%s) + outdeg(%s) + %s + ego(actors$node_type)",
            dn, nn, nn, ego_pc),
    sprintf("%s ~ indeg(%s) + outdeg(%s) + %s",
            dn, nn, nn, ego_pc),
    sprintf("%s ~ indeg(%s) + outdeg(%s)",
            dn, nn, nn),
    sprintf("%s ~ outdeg(%s)", dn, nn),
    sprintf("%s ~ indeg(%s)",  dn, nn)
  )
  for (i in seq_along(specs)) {
    res <- try_estimate(specs[[i]], "rate")
    if (!is.null(res)) { attr(res, "spec_used") <- specs[[i]]; return(res) }
  }
  NULL
}

get_named_coefs <- function(mod) {
  if (is.null(mod)) return(NULL)
  cv <- coef(mod); nm <- names(cv)
  if (is.null(nm) || any(nm == "")) nm <- paste0("c", seq_along(cv))
  cv <- as.numeric(cv); names(cv) <- nm
  cv
}

match_coef <- function(beta, pat) {
  if (is.null(beta)) return(NULL)
  idx <- grep(pat, names(beta))
  if (length(idx) == 0) return(NULL)
  beta[idx[1]]
}

## Positional lookup for sim / ego coefs (goldfish names them all "sim" or "ego")
get_coefs_by_name <- function(beta, nm) {
  if (is.null(beta)) return(numeric(0))
  idx <- which(names(beta) == nm)
  if (length(idx) == 0) return(numeric(0))
  beta[idx]
}

build_final_adj <- function(edge_types, type_objs, actors_df) {
  setNames(lapply(edge_types, function(t) {
    ev <- type_objs[[t]]$events
    m <- matrix(0, nrow = length(actors_df$label), ncol = length(actors_df$label),
                dimnames = list(actors_df$label, actors_df$label))
    if (nrow(ev) > 0) {
      for (k in seq_len(nrow(ev))) {
        m[ev$sender[k], ev$receiver[k]] <- m[ev$sender[k], ev$receiver[k]] + 1
      }
    }
    m
  }), edge_types)
}

## Scoring with features. Running adjacency so temporal eval is time-correct.
score_choice_running <- function(i, j, etype, adj_list, choice_coefs) {
  beta <- choice_coefs[[etype]]
  if (is.null(beta)) return(NA_real_)
  adj  <- adj_list[[etype]]
  i_in <- i %in% rownames(adj); j_in <- j %in% rownames(adj)
  if (!i_in && !j_in) return(NA_real_)
  
  lp <- 0
  ## Structural (need both endpoints)
  if (i_in && j_in) {
    if (!is.null(b <- match_coef(beta, "^inertia"))) lp <- lp + b * adj[i, j]
    if (!is.null(b <- match_coef(beta, "^recip")))   lp <- lp + b * adj[j, i]
    if (!is.null(b <- match_coef(beta, "^trans")))   lp <- lp + b * sum(adj[i, ] * adj[, j])
  }
  
  ## PCA similarity (positional)
  sim_betas <- get_coefs_by_name(beta, "sim")
  n_sim <- min(length(sim_betas), K_PCA)
  if (n_sim > 0) {
    for (k in seq_len(n_sim)) {
      pi_ <- pca_scores[i, k]; pj_ <- pca_scores[j, k]
      if (!is.na(pi_) && !is.na(pj_)) {
        lp <- lp + as.numeric(sim_betas[k]) * (-abs(pi_ - pj_))
      }
    }
  }
  
  ## same(node_type)
  b <- match_coef(beta, "^same")
  if (!is.null(b)) {
    ti <- node_type_lookup[i]; tj <- node_type_lookup[j]
    if (!is.na(ti) && !is.na(tj)) lp <- lp + b * as.numeric(ti == tj)
  }
  
  lp
}

score_rate_running <- function(i, etype, rate_coefs, running_indeg, running_outdeg) {
  beta <- rate_coefs[[etype]]
  if (is.null(beta)) return(NA_real_)
  if (!(i %in% names(running_indeg))) return(NA_real_)
  
  lp <- 0
  if (!is.null(b <- match_coef(beta, "^indeg")))  lp <- lp + b * running_indeg[i]
  if (!is.null(b <- match_coef(beta, "^outdeg"))) lp <- lp + b * running_outdeg[i]
  
  ## ego(pc_k)
  ego_betas <- get_coefs_by_name(beta, "ego")
  n_ego <- min(length(ego_betas), K_PCA)
  if (n_ego > 0) {
    for (k in seq_len(n_ego)) {
      pi_ <- pca_scores[i, k]
      if (!is.na(pi_)) lp <- lp + as.numeric(ego_betas[k]) * pi_
    }
  }
  
  lp
}

## ================== run_eval: ONE FULL EVALUATION ==================

run_eval <- function(events_train, events_test, split_name, seed) {
  cat("\n## SPLIT:", split_name, "| seed:", seed, "\n")
  cat("   Train:", nrow(events_train), "  Test:", nrow(events_test), "\n")
  
  ## Clear globalenv from previous run
  to_clear <- ls(envir = globalenv())
  to_clear <- to_clear[grepl("^(ev_|net_|dep_)", to_clear)]
  if (length(to_clear) > 0) rm(list = to_clear, envir = globalenv())
  
  ## Actors df with PCA + node_type
  actors_df <- data.frame(
    label   = sort(unique(c(events_train$from, events_train$to))),
    present = TRUE,
    stringsAsFactors = FALSE
  )
  actors_df$node_type <- node_type_lookup[actors_df$label]
  actors_df$node_type[is.na(actors_df$node_type) | actors_df$node_type == ""] <- "unk"
  for (j in seq_len(K_PCA)) {
    col <- paste0("pc", j)
    actors_df[[col]] <- pca_scores[actors_df$label, j]
    actors_df[[col]][is.na(actors_df[[col]])] <- 0
  }
  
  actors <- defineNodes(nodes = actors_df)
  assign("actors", actors, envir = globalenv())
  
  type_objs <- setNames(
    lapply(edge_types, function(t) build_type_objects(t, events_train, actors)),
    edge_types
  )
  
  cat("   Fitting choice ... ")
  choice_models <- setNames(lapply(edge_types, fit_choice), edge_types)
  cat(" rate ... ")
  rate_models   <- setNames(lapply(edge_types, fit_rate),   edge_types)
  cat("done\n")
  
  choice_coefs <- setNames(lapply(choice_models, get_named_coefs), edge_types)
  rate_coefs   <- setNames(lapply(rate_models,   get_named_coefs), edge_types)
  
  fitted_choice <- edge_types[!sapply(choice_coefs, is.null)]
  fitted_rate   <- edge_types[!sapply(rate_coefs,   is.null)]
  cat("   Fitted choice:", paste(fitted_choice, collapse = ","),
      " rate:", paste(fitted_rate, collapse = ","), "\n")
  
  final_adj <- build_final_adj(edge_types, type_objs, actors_df)
  
  ## ---------- TASKS 1 & 2 with running adjacency ----------
  ev_sorted  <- events_test %>% arrange(time_num)
  running_adj <- lapply(final_adj, function(m) m)
  
  exist_scores <- c(); exist_labels <- c(); n_exist_pos <- 0
  type_probs <- matrix(NA_real_, nrow = nrow(ev_sorted), ncol = length(fitted_choice),
                       dimnames = list(NULL, fitted_choice))
  
  train_node_set <- actors_df$label
  
  for (k in seq_len(nrow(ev_sorted))) {
    s <- ev_sorted$from[k]; r <- ev_sorted$to[k]; et <- ev_sorted$type[k]
    
    ## Task 1
    if (et %in% fitted_choice) {
      pos <- score_choice_running(s, r, et, running_adj, choice_coefs)
      if (!is.na(pos)) {
        n_exist_pos <- n_exist_pos + 1
        cand <- setdiff(train_node_set, c(s, r))
        if (length(cand) > 0) {
          neg_ids <- sample(cand, min(K_NEG, length(cand)))
          neg <- vapply(neg_ids, function(nn) {
            ## swap the train-side node
            if (s %in% train_node_set && !(r %in% train_node_set))
              score_choice_running(s, nn, et, running_adj, choice_coefs)
            else if (r %in% train_node_set && !(s %in% train_node_set))
              score_choice_running(nn, r, et, running_adj, choice_coefs)
            else
              score_choice_running(s, nn, et, running_adj, choice_coefs)
          }, numeric(1))
          exist_scores <- c(exist_scores, pos, neg)
          exist_labels <- c(exist_labels, 1, rep(0, length(neg)))
        }
      }
    }
    
    ## Task 2
    raw <- vapply(fitted_choice,
                  function(e2) score_choice_running(s, r, e2, running_adj, choice_coefs),
                  numeric(1))
    if (!any(is.na(raw))) {
      raw <- raw - max(raw)
      type_probs[k, ] <- exp(raw) / sum(exp(raw))
    }
    
    ## update adj after scoring
    if (et %in% edge_types &&
        s %in% rownames(running_adj[[et]]) &&
        r %in% rownames(running_adj[[et]])) {
      running_adj[[et]][s, r] <- running_adj[[et]][s, r] + 1
    }
  }
  
  ## Task 1 AUC
  ok <- !is.na(exist_scores)
  auc_exist <- NA_real_
  if (sum(ok) > 0 && length(unique(exist_labels[ok])) == 2 &&
      var(exist_scores[ok]) > 0) {
    auc_exist <- as.numeric(auc(
      roc(exist_labels[ok], exist_scores[ok], quiet = TRUE)
    ))
  }
  
  ## Task 2 metrics
  fit_mask <- ev_sorted$type %in% fitted_choice
  ok_rows  <- complete.cases(type_probs) & fit_mask
  macro_auc <- NA_real_; top1 <- NA_real_; maj_baseline <- NA_real_
  per_type_auc <- setNames(rep(NA_real_, length(edge_types)), edge_types)
  if (sum(ok_rows) > 10) {
    ovr <- c()
    for (et in fitted_choice) {
      y <- as.integer(ev_sorted$type[ok_rows] == et)
      if (length(unique(y)) == 2 && var(type_probs[ok_rows, et]) > 0) {
        a <- as.numeric(auc(roc(y, type_probs[ok_rows, et], quiet = TRUE)))
        ovr <- c(ovr, setNames(a, et))
        per_type_auc[et] <- a
      }
    }
    if (length(ovr) > 0) macro_auc <- mean(ovr)
    pred_class <- fitted_choice[max.col(type_probs[ok_rows, , drop = FALSE])]
    top1 <- mean(pred_class == ev_sorted$type[ok_rows])
    maj_baseline <- max(table(ev_sorted$type[ok_rows])) / sum(ok_rows)
  }
  
  ## ---------- TASK 3 ----------
  running_indeg <- setNames(lapply(edge_types, function(t) {
    adj <- final_adj[[t]]
    setNames(rep(0, nrow(adj)), rownames(adj))
  }), edge_types)
  running_outdeg <- setNames(lapply(edge_types, function(t) {
    adj <- final_adj[[t]]
    setNames(rep(0, nrow(adj)), rownames(adj))
  }), edge_types)
  
  dyn_lp <- numeric(nrow(ev_sorted))
  for (k in seq_len(nrow(ev_sorted))) {
    s <- ev_sorted$from[k]; r <- ev_sorted$to[k]; et <- ev_sorted$type[k]
    if (!(et %in% fitted_rate)) { dyn_lp[k] <- NA_real_; next }
    actor <- if (s %in% names(running_indeg[[et]])) s
    else if (r %in% names(running_indeg[[et]])) r
    else NA
    if (is.na(actor)) { dyn_lp[k] <- NA_real_; next }
    dyn_lp[k] <- score_rate_running(actor, et, rate_coefs,
                                    running_indeg[[et]], running_outdeg[[et]])
    if (s %in% names(running_outdeg[[et]])) running_outdeg[[et]][s] <- running_outdeg[[et]][s] + 1
    if (r %in% names(running_indeg[[et]]))  running_indeg[[et]][r]  <- running_indeg[[et]][r]  + 1
  }
  ev_sorted$dyn_lp <- dyn_lp
  ev_dyn <- ev_sorted[!is.na(ev_sorted$dyn_lp), ]
  
  spearman_global <- NA_real_; pair_acc <- NA_real_; rank_mae_n <- NA_real_
  wg_mean_rho <- NA_real_; wg_weighted_rho <- NA_real_
  if (nrow(ev_dyn) > 2 && var(ev_dyn$dyn_lp) > 0) {
    spearman_global <- suppressWarnings(
      cor(ev_dyn$dyn_lp, ev_dyn$time_num, method = "spearman")
    )
    
    wg <- ev_dyn %>%
      group_by(from, type) %>%
      filter(n() >= 2, var(dyn_lp) > 0, var(time_num) > 0) %>%
      summarise(rho = suppressWarnings(cor(dyn_lp, time_num, method = "spearman")),
                n = n(), .groups = "drop") %>%
      filter(!is.na(rho))
    if (nrow(wg) > 0) {
      wg_mean_rho     <- mean(wg$rho)
      wg_weighted_rho <- sum(wg$rho * wg$n) / sum(wg$n)
    }
    
    n <- nrow(ev_dyn)
    np <- min(50000, n * (n - 1) / 2)
    i1 <- sample.int(n, np, replace = TRUE); i2 <- sample.int(n, np, replace = TRUE)
    keep <- i1 != i2; i1 <- i1[keep]; i2 <- i2[keep]
    ts <- sign(ev_dyn$time_num[i1] - ev_dyn$time_num[i2])
    ls <- sign(ev_dyn$dyn_lp[i2]   - ev_dyn$dyn_lp[i1])
    keep2 <- ts != 0 & ls != 0
    pair_acc <- mean(ts[keep2] == ls[keep2])
    
    tr <- rank(ev_dyn$time_num, ties.method = "average")
    pr <- rank(-ev_dyn$dyn_lp,  ties.method = "average")
    rank_mae_n <- mean(abs(tr - pr)) / n
  }
  
  cat(sprintf("   T1 AUC=%.3f  T2 macro=%.3f top1=%.3f  T3 |rho|=%.3f wg=%.3f\n",
              auc_exist, macro_auc, top1, abs(spearman_global), wg_mean_rho))
  
  list(
    split                   = split_name,
    seed                    = seed,
    n_train                 = nrow(events_train),
    n_test                  = nrow(events_test),
    n_fitted_choice         = length(fitted_choice),
    n_fitted_rate           = length(fitted_rate),
    task1_roc_auc           = auc_exist,
    task2_macro_auc         = macro_auc,
    task2_top1              = top1,
    task2_majority_baseline = maj_baseline,
    task2_auc_catalysis     = unname(per_type_auc["catalysis"]),
    task2_auc_complex       = unname(per_type_auc["complex_component"]),
    task2_auc_expression    = unname(per_type_auc["expression"]),
    task2_auc_reaction      = unname(per_type_auc["reaction"]),
    task2_auc_translocation = unname(per_type_auc["translocation"]),
    task3_spearman          = spearman_global,
    task3_spearman_abs      = abs(spearman_global),
    task3_pair_accuracy     = pair_acc,
    task3_rank_mae_norm     = rank_mae_n,
    task3_wg_mean_rho       = wg_mean_rho,
    task3_wg_weighted_rho   = wg_weighted_rho
  )
}

## ================== PREPARE TEMPORAL SPLIT (deterministic) ==================

unique_times <- sort(unique(events$time_num))
cum_events   <- cumsum(tabulate(match(events$time_num, unique_times)))
target_count <- floor(0.8 * nrow(events))
cut_step_idx <- which(cum_events >= target_count)[1]
cut_time     <- unique_times[cut_step_idx]

events_train_T <- events %>% filter(time_num <= cut_time)
events_test_T  <- events %>% filter(time_num >  cut_time)

cat("Temporal cut at time =", cut_time, "\n")
cat("  Train:", nrow(events_train_T), "  Test:", nrow(events_test_T), "\n")

## ================== RUN ALL SEEDS × BOTH SPLITS ==================

all_results <- list()

for (seed in SEEDS) {
  set.seed(seed)
  
  ## Temporal split: data split is deterministic; only scoring randomness (negatives) varies
  res_T <- run_eval(events_train_T, events_test_T, "temporal", seed)
  all_results[[length(all_results) + 1]] <- res_T
  
  ## Semi-inductive split: re-sample unseen nodes per seed
  set.seed(seed)
  all_nodes    <- unique(c(events$from, events$to))
  unseen_nodes <- sample(all_nodes, size = floor(0.2 * length(all_nodes)))
  seen_nodes   <- setdiff(all_nodes, unseen_nodes)
  
  events_train_S <- events %>% filter(from %in% seen_nodes, to %in% seen_nodes)
  events_test_S  <- events %>%
    filter(
      (from %in% unseen_nodes & to %in% seen_nodes) |
        (from %in% seen_nodes   & to %in% unseen_nodes)
    )
  
  res_S <- run_eval(events_train_S, events_test_S, "semi_inductive", seed)
  all_results[[length(all_results) + 1]] <- res_S
}

## ================== ASSEMBLE RESULTS ==================

results_df <- bind_rows(lapply(all_results, as_tibble))

raw_path <- "../.cache/rem_results_raw_se.csv"
write_csv(results_df, raw_path)
cat("\nRaw per-seed results saved to:", raw_path, "\n")

## Summary: mean, sd, n_valid per metric per split
numeric_cols <- c("task1_roc_auc", "task2_macro_auc", "task2_top1",
                  "task2_majority_baseline",
                  "task2_auc_catalysis", "task2_auc_complex",
                  "task2_auc_expression", "task2_auc_reaction",
                  "task2_auc_translocation",
                  "task3_spearman", "task3_spearman_abs",
                  "task3_pair_accuracy", "task3_rank_mae_norm",
                  "task3_wg_mean_rho", "task3_wg_weighted_rho")

summary_df <- results_df %>%
  group_by(split) %>%
  summarise(across(all_of(numeric_cols),
                   list(
                     mean = ~ mean(.x, na.rm = TRUE),
                     sd   = ~ sd(.x,   na.rm = TRUE),
                     n    = ~ sum(!is.na(.x))
                   ),
                   .names = "{.col}__{.fn}"),
            .groups = "drop")

summary_path <- "../.cache/rem_results_summary_se.csv"
write_csv(summary_df, summary_path)
cat("Summary saved to:", summary_path, "\n")

## Pretty-print a paper-style table
cat("\n##############################################\n")
cat("FINAL RESULTS (mean +/- sd over", length(SEEDS), "seeds)\n")
cat("##############################################\n\n")

fmt <- function(x_mean, x_sd, digits = 3) {
  if (is.na(x_mean)) return("NA")
  sprintf("%.*f \u00b1 %.*f", digits, x_mean, digits, x_sd)
}

for (sp in unique(summary_df$split)) {
  row <- summary_df[summary_df$split == sp, ]
  cat("-- SPLIT:", sp, "--\n")
  cat("  Task 1 ROC-AUC            :",
      fmt(row$task1_roc_auc__mean, row$task1_roc_auc__sd), "\n")
  cat("  Task 2 macro-AUC          :",
      fmt(row$task2_macro_auc__mean, row$task2_macro_auc__sd), "\n")
  cat("  Task 2 top-1              :",
      fmt(row$task2_top1__mean, row$task2_top1__sd), "\n")
  cat("  Task 2 majority baseline  :",
      fmt(row$task2_majority_baseline__mean, row$task2_majority_baseline__sd), "\n")
  cat("  Task 3 Spearman (signed)  :",
      fmt(row$task3_spearman__mean, row$task3_spearman__sd), "\n")
  cat("  Task 3 |Spearman|         :",
      fmt(row$task3_spearman_abs__mean, row$task3_spearman_abs__sd), "\n")
  cat("  Task 3 pair accuracy      :",
      fmt(row$task3_pair_accuracy__mean, row$task3_pair_accuracy__sd), "\n")
  cat("  Task 3 rank MAE/N         :",
      fmt(row$task3_rank_mae_norm__mean, row$task3_rank_mae_norm__sd), "\n")
  cat("  Task 3 within-group rho   :",
      fmt(row$task3_wg_mean_rho__mean, row$task3_wg_mean_rho__sd), "\n")
  cat("  Task 3 weighted wg rho    :",
      fmt(row$task3_wg_weighted_rho__mean, row$task3_wg_weighted_rho__sd), "\n\n")
}

cat("Done.\n")
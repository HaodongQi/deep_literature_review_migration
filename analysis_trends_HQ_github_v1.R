
# If running in RStudio, set working dir to script location (optional)
if (requireNamespace("rstudioapi", quietly = TRUE) &&
  rstudioapi::isAvailable()) {
    setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
  }
pacman::p_load(data.table, tidyverse, ggplot2)

# -- recode function
recodeFun <- function(df){
  df <- df |> as.data.frame()
  # for Q2
  df <- df %>% mutate(Q2=ifelse(Q2=="", "999", Q2))
  # FOR Q3
  df <- df %>% mutate(Q3=ifelse(Q3=="", "999", Q3))
  # FOR Q4
  df <- df %>% mutate(Q4=ifelse(Q4=="", "999", Q4))
  # %>%
  #   mutate(Q4=lapply(Q4, function(x) if(x=="0") x="42" else x=x) ) # None=0; Other=42
  # FOR Q5
  df <- df %>% mutate(Q5=ifelse(Q5=="", "999", Q5) )
}

# master data
temp <-  fread("wos_migration_master.csv", header = T) %>% 
  as_tibble() %>% select(DOI, "Article Title", "Publication Year", CI)
fullMap <- fread("wos_migration_mapped.csv", header = T) %>% 
  as_tibble() %>% select(DOI, "Article Title", id) %>% 
  left_join(., temp, by=c("DOI", "Article Title")) %>% 
  dplyr::rename(Title="Article Title",PubYear=`Publication Year`) %>% 
  select(id, PubYear, CI, Title)
fullTestFT <- fread("classifications-wos-100.csv", colClasses = "character", fill=TRUE) %>% 
  as_tibble() 
fullTestFT <- recodeFun(fullTestFT)
  # condition data on Q1==1 (is migration study), remove Q5 discipline
# fullTestFT <- fullTestFT %>% filter(Q1==1)

# map key values
fullTestFT <- fullTestFT %>% 
  mutate(is.mig=ifelse(Q1==1,1,0)) %>% 
  gather(Question, value, -id, -is.mig)
keyValue <- fread("key-values.csv", colClasses = "character", fill=TRUE) %>% 
  as_tibble() %>% dplyr::rename(Question=question)

fullTestFT <- fullTestFT %>% separate_rows(value, sep = " ") %>% 
  left_join(keyValue, by=c("value", "Question")) %>% 
  mutate(key=ifelse(value=="999", "none", key) ) %>% 
  left_join(.,fullMap)

# label questions
fullTestFT <- fullTestFT %>% mutate(i=1) %>% 
  mutate(qLabel=case_when(
    Question=="Q1" ~ "1. About migration and mobility?",
    Question=="Q2" ~ "3. What migration drivers discussed?",
    Question=="Q3" ~ "Sentiment",
    Question=="Q4" ~ "4. What clim. and env. hazards discussed?",
    Question=="Q5" ~ "Discipline",
    Question=="Q6" ~ "2. Used quant. methods?",
    TRUE ~ Question)) %>% 
  mutate(key=case_when(
    grepl("migration drivers",qLabel) & grepl("enviro", key) ~ "env.|clim.",
    grepl("migration drivers",qLabel) & grepl("clim", key) ~ "env.|clim.",
    TRUE  ~ key
  ))
write.csv(fullTestFT, "fullSetInference.csv", row.names = F)


# plot trends conditional on mig studies
pacman::p_load(geomtextpath)
plotdf <- read.csv("fullSetInference.csv") %>% 
  filter(!grepl("Senti",qLabel) & !grepl("Disc",qLabel) #& PubYear>=1990
         #& !grepl("none",key) 
         ) 

sel <- plotdf |> select(id, PubYear) |> distinct()
temp <- plotdf %>% filter(grepl("About mig", qLabel)) %>%   
  right_join(sel, by = c("id", "PubYear")) |> 
  group_by(qLabel,key,PubYear) %>% 
  dplyr::summarise(N.Labels=sum(i)) 
temp_tab <- temp |> group_by(qLabel, key) |> 
  dplyr::summarise(N.Labels=sum(N.Labels)) |> 
  mutate(N.Pubs=length(unique(sel$id)))
temp <- ggplot(data=temp, 
             aes(x=PubYear,y=N.Labels, color=key)) +
  facet_wrap(.~qLabel, ncol=1, scale="free_y") +
  geom_textline(aes(label=key), show.legend=F,  alpha=.8, 
    linewidth = 0.5, gap = TRUE, # leaves a gap under the text
    vjust = .8, hjust=.9) +  theme_bw()
p1 <- temp; p1_tab <- temp_tab

sel <- plotdf |> filter(is.mig==1) |> select(id, PubYear) |> distinct()
temp <- plotdf %>% filter(grepl("methods", qLabel)) %>% 
  right_join(sel, by = c("id", "PubYear")) |> 
  group_by(qLabel,key,PubYear) %>% 
  dplyr::summarise(N.Labels=sum(i)) 
temp_tab <- temp |> group_by(qLabel, key) |> 
  dplyr::summarise(N.Labels=sum(N.Labels)) |> 
  mutate(N.Pubs=length(unique(sel$id)))
temp <- ggplot(data=temp, 
             aes(x=PubYear,y=N.Labels, color=key)) +
  facet_wrap(.~qLabel, nrow=1, scale="free_y") +
  geom_textline(aes(label=key), show.legend=F,  alpha=.8, 
    linewidth = 0.5, gap = TRUE, # leaves a gap under the text
    vjust = .8, hjust=.9) +  theme_bw()
p2 <- temp; p2_tab <- temp_tab

sel <- plotdf |> filter(is.mig==1) |> select(id, PubYear) |> distinct()
temp <- plotdf %>% filter(grepl("driver", qLabel, ignore.case=T)) %>% 
  right_join(sel, by = c("id", "PubYear")) |> 
  group_by(qLabel,key,PubYear) %>% 
  dplyr::summarise(N.Labels=sum(i)) 
temp_lab <- temp |> group_by(key) |> filter(PubYear==max(PubYear))
temp_tab <- temp |> group_by(qLabel, key) |> 
  dplyr::summarise(N.Labels=sum(N.Labels)) |> 
  mutate(N.Pubs=length(unique(sel$id)))
temp <- ggplot(data=temp, 
             aes(x=PubYear,y=N.Labels, color=key)) +
  facet_wrap(.~qLabel, nrow=1, scale="free_y") +
  geom_line(aes(color=key), show.legend = F) +
  ggrepel::geom_text_repel( data = temp_lab,aes(label=key), show.legend=F) +
  theme_bw()
p3 <- temp; p3_tab <- temp_tab

sel <- plotdf |> filter(grepl("drivers",qLabel)  & grepl("env.", key) & is.mig==1) |> 
  select(id, PubYear) |> distinct()
temp <- plotdf |> filter(grepl("hazard", qLabel) ) %>% 
  right_join(sel, by = c("id", "PubYear")) |> 
  group_by(qLabel,key,PubYear) %>% 
  dplyr::summarise(N.Labels=sum(i)) 
temp_lab <- temp |> group_by(key) |> filter(PubYear==max(PubYear))
temp_tab <- temp |> group_by(qLabel, key) |> 
  dplyr::summarise(N.Labels=sum(N.Labels)) |> 
  mutate(N.Pubs=length(unique(sel$id)))
temp <- ggplot(data=temp, 
             aes(x=PubYear,y=N.Labels, color=key)) +
  facet_wrap(.~qLabel, nrow=1, scale="free_y") +
  geom_line(aes(color=key), show.legend = F) +
  ggrepel::geom_text_repel( data = temp_lab,aes(label=key), show.legend=F) +
  theme_bw()
p4 <- temp; p4_tab <- temp_tab

pacman::p_load(cowplot)
plot_grid(p1, p2, p3, p4, ncol = 2, labels = c("a", "b", "c", "d"))
ggsave("trends.png", width = 10, height = 8)

# -- tab n labels
temp <- rbind(p1_tab,p2_tab,p3_tab,p4_tab) |> 
  dplyr::rename(Questions=qLabel, Labels=key)

pacman::p_load(knitr)

# 1) Build the multirow cells (as in your pipeline), and ensure groups are contiguous
temp_grouped <- temp %>%
  arrange(Questions, N.Pubs, N.Labels) %>%   # keep groups together
  group_by(Questions, N.Pubs) %>%
  mutate(
    n_rows = n(),
    Questions_cell = ifelse(
      dplyr::row_number() == 1,
      sprintf("\\multirow{%d}{*}{%s}", n_rows, Questions),
      ""
    ),
    N.Pubs_cell = ifelse(
      dplyr::row_number() == 1,
      sprintf("\\multirow{%d}{*}{%d}", n_rows, N.Pubs),
      ""
    )
  ) %>%
  ungroup()

# 2) Prepare the data frame to print
temp_out <- temp_grouped %>%
  select(
    Questions = Questions_cell,
    Labels,
    N.Labels,
    N.Pubs = N.Pubs_cell
  )

# 3) Build a linesep vector that inserts \midrule **after** each group
#    We use the original (non-_cell) columns to detect group ends.
group_ends <- temp_grouped %>%
  group_by(Questions, N.Pubs) %>%
  mutate(is_last_in_group = row_number() == n()) %>%
  ungroup() %>%
  pull(is_last_in_group)

# linesep must be a character vector of length nrow(temp_out)
# Put a \midrule after the last row of each group, otherwise nothing.
linesep_vec <- ifelse(group_ends, "\\midrule", "")

# 4) Print LaTeX table: no addlinespace, \midrule between questions
kable(
  temp_out,
  format = "latex",
  booktabs = TRUE,
  escape = FALSE,  # allow \multirow to pass through
  col.names = c("Questions", "Labels", "N.Labels", "N.Pubs"),
  align = c("l","l","r","r"),
  linesep = linesep_vec
)

#--- heatmaps
# Load necessary libraries
pacman::p_load(viridis,pheatmap,grid,RColorBrewer)

# Read the dataset
df <- read.csv("fullSetInference.csv")

# Filter Data 
migration_drivers <- df %>%
  filter(grepl("migration drivers",qLabel)  & grepl("env.", key)) %>%
  select(id, PubYear)

hazards_data <- df %>%
  filter(grepl("clim",qLabel)) %>%
  select(id, PubYear, key, i)

# Merge migration drivers with hazards
hazards_data <- hazards_data %>%
  right_join(migration_drivers, by = c("id", "PubYear")) %>%
  filter(key != "none") 

# Hazard categories
hazards_data <- hazards_data %>%
  mutate(
    hazard_category = case_when(
      # Climatological Hazard
      key %in% c("ocean acidification", "sea ice", "acid rain", "drought",
                 "cold wave", "heat wave", "glacial", "wildfire") ~ "Climatological Hazard",
      
      # Environmental Degradation
      key %in% c("air pollution", "biodiversity loss", "coastal erosion", "land degradation",
                 "deforestation", "desertification", "water pollution", "diebacks",
                 "permafrost loss", "sand mining", "sea level rise") ~ "Environmental Degradation",
      
      # Geohazards Hazard
      key %in% c("earthquake", "tsunami", "volcanic", "avalanche", "landslide") ~ "Geohazards Hazard",
      
      # Hydrological Hazard
      key %in% c("flood") ~ "Hydrological Hazard",
      
      # Meteorological Hazard
      key %in% c("downburst", "thunderstorm", "sandstorm", "haze", "storm", "cyclone", 
                 "blizzard", "hail", "snow", "tornado", "wind", "hurricane", "typhoon") ~ "Meteorological Hazard",
      
      # Biological Hazard
      key %in% c("algal blooms", "infestation", "infectious diseases") ~ "Biological Hazard",
      
      # Default category (if anything is missing)
      TRUE ~ "Other"
    )
  )
hazards_data <- hazards_data %>% distinct()

# Aggregate hazard occurrences per year
plotdf_hazards <- hazards_data %>%
  group_by(PubYear, key) %>%
  summarise(N.Pubs = sum(i, na.rm = TRUE), .groups = 'drop') %>%
# Find global min/max once
  mutate(.min_year = min(PubYear), .max_year = max(PubYear)) %>%
  complete(
    key,
    PubYear = seq(min(.min_year), max(.max_year)),
    fill = list(N.Pubs = 0)
  ) %>%
  select(-.min_year, -.max_year) %>%
  arrange(key, PubYear) |> 
  pivot_wider(names_from = PubYear, values_from = N.Pubs, values_fill = 0) %>%
  column_to_rownames("key")

# Clustering
distance_matrix <- dist(plotdf_hazards, method = "euclidean")
clustering <- hclust(distance_matrix, method = "ward.D2")

# Coloring
color_palette <- colorRampPalette((brewer.pal(10, "YlOrRd")))(80)

# Heatmap with Dendro
png("hazHeatmap.png", width = 800, height = 500)  
pheatmap(
  plotdf_hazards,
  clustering_rows = clustering,   
  clustering_cols = FALSE,        
  cluster_cols = FALSE,          
  color = color_palette,          
  fontsize_row = 13,
  fontsize_col = 13,
  angle_col = "90"
)
dev.off()


# --- network analysis of hazard
library(pacman)
pacman::p_load(dplyr, tidyverse, ggplot2, igraph, ggraph, tidygraph, patchwork)

filtered_data <- hazards_data %>%
  filter(!key %in% c("none")) 

years <- sort(unique(filtered_data$PubYear))
all_hazards <- sort(unique(filtered_data$key))

# network
full_hazards_by_doc <- filtered_data %>%
  group_by(id) %>%
  summarise(hazards = list(unique(key)))

A_total <- matrix(0, nrow = length(all_hazards), ncol = length(all_hazards),
                  dimnames = list(all_hazards, all_hazards))

for (hazard_list in full_hazards_by_doc$hazards) {
  if (length(hazard_list) >= 2) {
    pairs <- combn(hazard_list, 2)
    for (i in 1:ncol(pairs)) {
      h1 <- pairs[1, i]
      h2 <- pairs[2, i]
      A_total[h1, h2] <- A_total[h1, h2] + 1
      A_total[h2, h1] <- A_total[h2, h1] + 1
    }
  }
}

# fixed location for node 
g_total <- graph_from_adjacency_matrix(A_total, mode = "undirected", weighted = TRUE)
layout_fixed <- create_layout(as_tbl_graph(g_total), layout = "fr")
# Get consistent x/y axis limits
xlim_vals <- range(layout_fixed$x)*1.2
ylim_vals <- range(layout_fixed$y)*1.2

# fuction for network
create_network_plot <- function(year_data, title_label, layout_fixed) {
  if (nrow(year_data) < 2) return(NULL)
  
  hazards_by_doc <- year_data %>%
    group_by(id) %>%
    summarise(hazards = list(unique(key)))
  
  hazard_list <- sort(unique(year_data$key))
  
  A <- matrix(0, nrow = length(hazard_list), ncol = length(hazard_list),
              dimnames = list(hazard_list, hazard_list))
  
  for (hazards in hazards_by_doc$hazards) {
    if (length(hazards) >= 2) {
      pairs <- combn(hazards, 2)
      for (i in 1:ncol(pairs)) {
        h1 <- pairs[1, i]
        h2 <- pairs[2, i]
        A[h1, h2] <- A[h1, h2] + 1
        A[h2, h1] <- A[h2, h1] + 1
      }
    }
  }
  
  g <- graph_from_adjacency_matrix(A, mode = "undirected", weighted = TRUE)
  if (ecount(g) < 1 || vcount(g) < 2) return(NULL)
  
  clusters <- cluster_louvain(g)
  tg <- as_tbl_graph(g) %>% mutate(cluster = clusters$membership)
  
  node_names <- V(g)$name
  current_nodes <- layout_fixed %>% filter(name %in% node_names)
  # current_nodes <- current_nodes[match(node_names, current_nodes$name), ]
  
  # nudge_y adjustment not to overlap (hurricane, earthquake)
  p <- ggraph(tg, layout = "manual", x = current_nodes$x, y = current_nodes$y) +
    geom_edge_link(aes(width = weight), 
                   color = "gray80", show.legend = FALSE) +
    geom_node_point(aes(color = as.factor(cluster), size = centrality_degree(),
                        alpha=centrality_degree()), show.legend = FALSE) +
    geom_node_text(aes(label = name, alpha=centrality_degree()), 
                   size = 2.8, repel = T, show.legend = F) +
    scale_size_continuous(range = c(2, 6)) +
    scale_color_brewer(palette = "Set2") +
    coord_cartesian(xlim = xlim_vals, ylim = ylim_vals, expand = FALSE) +
    theme_void() +
    theme(
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.6),
      plot.background = element_rect(fill = NA, color = NA),
      plot.title = element_text(size = 11, hjust = 0.5),
      plot.margin = margin(1, 2, 1, 2) ) +
    labs(title = title_label)
  
  return(p)
}

# 5 years grouping
group_breaks <- c(1980, seq(1995, 2025, by = 5))
plot_list <- list()

for (i in seq_along(group_breaks)) {
  start_year <- group_breaks[i]
  end_year <- min(group_breaks[i+1]-1, max(years), na.rm=T)  
  
  group_data <- filtered_data %>%
    filter(PubYear >= start_year, PubYear <= end_year)

  n_pub <- length(unique(group_data$id))
  group_label <- paste0(start_year, "-", end_year, " (N_Pubs=", n_pub, ")")
  
  p <- create_network_plot(group_data, group_label, layout_fixed)
  if (!is.null(p)) {
    plot_list[[group_label]] <- p
  }
}

# Add plot for all years
clusters_total <- cluster_louvain(g_total)
tg_total <- as_tbl_graph(g_total) %>% mutate(cluster = clusters_total$membership)

p_total <- ggraph(tg_total, layout = "manual", x = layout_fixed$x, y = layout_fixed$y) +
  geom_edge_link(aes(width = weight), 
                 color = "gray80", show.legend = FALSE) +
  geom_node_point(aes(color = as.factor(cluster), size = centrality_degree(),
                      alpha=centrality_degree()), show.legend = FALSE) +
  geom_node_text(aes(label = name, alpha=centrality_degree()), 
                 size = 2.8, repel = T, show.legend = F) +
  scale_size_continuous(range = c(2, 6)) +
  scale_color_brewer(palette = "Set2") +
  coord_cartesian(xlim = xlim_vals, ylim = ylim_vals, expand = FALSE) +
  theme_void() +
  theme(
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.6),
    plot.background = element_rect(fill = NA, color = NA),
    plot.title = element_text(size = 11, hjust = 0.5),
    plot.margin = margin(1, 2, 1, 2) ) +
  labs(title = paste0(
    "All Years", " (N_Pubs=",length(unique(full_hazards_by_doc$id)),")"))

# combine all plots
plot_list[["All Years"]] <- p_total

final_plot <- wrap_plots(plot_list, ncol = 2)

ggsave("hazard_network_grouped_5year.png", final_plot, 
       width = 8, height = 10, dpi = 1000)




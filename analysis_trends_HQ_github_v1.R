


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
    Question=="Q1" ~ "1. About migration & mobility?",
    Question=="Q2" ~ "3. What migration drivers discussed?",
    Question=="Q3" ~ "Sentiment",
    Question=="Q4" ~ "4. What clim. & env. hazards discussed?",
    Question=="Q5" ~ "Discipline",
    Question=="Q6" ~ "2. Used quant. methods?",
    TRUE ~ Question)) %>% 
  mutate(key=case_when(
    grepl("migration drivers",qLabel) & grepl("enviro", key) ~ "env.|clim.",
    grepl("migration drivers",qLabel) & grepl("clim", key) ~ "env.|clim.",
    TRUE  ~ key
  ))
write.csv(fullTestFT, "fullSetInference.csv", row.names = F)

# # plot trends
# plotdf <- read.csv("fullSetInference.csv") %>% 
#   group_by(qLabel,key,PubYear) %>% 
#   dplyr::summarise(N.Pubs=sum(i)) 
# plotdf <- plotdf %>% 
#   filter(!grepl("Senti",qLabel) & !grepl("Disc",qLabel) & PubYear>=1990
#          #& !grepl("none",key)
#          ) 
# labeldf <- plotdf %>% group_by(qLabel,key) %>% 
#   dplyr::summarise(max.x=max(PubYear), max.y=max(N.Pubs))
# 
# ggplot(plotdf, aes(x=PubYear,y=N.Pubs, color=key)) +
#   facet_wrap(.~qLabel, ncol=1, scale="free_y"
#              ) +
#   geom_line(show.legend = F, linetype="longdash",alpha=.6) +
#   ggrepel::geom_text_repel(
#     data=labeldf,aes(max.x, max.y, label=key, size=max.y), 
#     show.legend = F, max.overlaps=55) +
#   theme_bw()
# ggsave("trends.png", width = 8, height = 8)

# plot trends conditional on mig studies
pacman::p_load(geomtextpath)
plotdf <- read.csv("fullSetInference.csv") %>% 
  filter(!grepl("Senti",qLabel) & !grepl("Disc",qLabel) & PubYear>=1990
         #& !grepl("none",key) 
         ) 
plotdf %>% filter(grepl("driver",qLabel)) %>% select(key) %>% unique()         

p1 <- plotdf %>% filter(grepl("About mig", qLabel)) %>% 
  group_by(qLabel,key,PubYear) %>% 
  dplyr::summarise(N.Pubs=sum(i)) 
labeldf <- p1 %>% group_by(qLabel,key) %>% 
  dplyr::summarise(max.x=max(PubYear), max.y=max(N.Pubs))
p1 <- ggplot(data=p1, 
             aes(x=PubYear,y=N.Pubs, color=key)) +
  facet_wrap(.~qLabel, ncol=1, scale="free_y"
  ) +
  # geom_line(show.legend = F, linetype="longdash",alpha=.8) +
  # ggrepel::geom_text_repel(
  #   data=labeldf,aes(max.x, max.y, label=key), 
  #   show.legend = F, max.overlaps=55) +
  # geom_text(
  #   data=labeldf,aes(max.x, max.y, label=key), 
  #   show.legend = F, check_overlap = T) +
  geom_textline(aes(label=key), show.legend=F, hjust=.8, vjust=.8, alpha=.8) +
  theme_bw()

p2 <- plotdf %>% filter(!grepl("About mig", qLabel) & is.mig==1) %>% 
  group_by(qLabel,key,PubYear) %>% 
  dplyr::summarise(N.Pubs=sum(i)) 
labeldf <- p2 %>% group_by(qLabel,key) %>% 
  dplyr::summarise(max.x=max(PubYear), max.y=max(N.Pubs))
p2 <- ggplot(data=p2, 
             aes(x=PubYear,y=N.Pubs, color=key)) +
  facet_wrap(.~qLabel, nrow=1, scale="free_y"
  ) +
  # geom_line(show.legend = F, linetype="longdash",alpha=.8) +
  # ggrepel::geom_text_repel(
  #   data=labeldf,aes(max.x, max.y, label=key, size=max.y), 
  #   show.legend = F, max.overlaps=55) +
  # geom_text(
  #   data=labeldf,aes(max.x-3, max.y, label=key, size=max.y*.6), 
  #   show.legend = F, check_overlap = T) +
  geom_textline(aes(label=key), show.legend=F, hjust=.8, vjust=.8, alpha=.8) +
  theme_bw()

pacman::p_load(cowplot)
plot_grid(p1, p2, ncol = 1, rel_heights = c(2,3), labels = c("a", "b"))
ggsave("trends.png", width = 9, height = 9)


#-- heatmaps
# Load necessary libraries
pacman::p_load(viridis,pheatmap,grid,RColorBrewer)

# Read the dataset
df <- read.csv("fullSetInference.csv")

# Filter Data 
migration_drivers <- df %>%
  filter(grepl("migration drivers",qLabel)  & grepl("env.", key)) %>%
  select(id, PubYear)

hazards_data <- df %>%
  filter(grepl("clim. & env. hazards ",qLabel)) %>%
  select(id, PubYear, key, i)

# Merge migration drivers with hazards
hazards_data <- hazards_data %>%
  inner_join(migration_drivers, by = c("id", "PubYear")) %>%
  filter(PubYear >= 1990 & key != "none")

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


# network analysis of hazard
library(pacman)
pacman::p_load(dplyr, tidyverse, ggplot2, igraph, ggraph, tidygraph, patchwork)

filtered_data <- merged_data %>%
  filter(!key %in% c("none")) %>%
  filter(PubYear >= 1990)

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
group_breaks <- seq(min(years), max(years), by = 5)
plot_list <- list()

for (i in seq_along(group_breaks)) {
  start_year <- group_breaks[i]
  end_year <- min(start_year + 4, max(years))  
  group_label <- paste0(start_year, "-", end_year)
  
  group_data <- filtered_data %>%
    filter(PubYear >= start_year, PubYear <= end_year)
  
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
  labs(title = "All Years")

# combine all plots
plot_list[["All Years"]] <- p_total

final_plot <- wrap_plots(plot_list, ncol = 2)

ggsave("hazard_network_grouped_5year.png", final_plot, 
       width = 8, height = 10, dpi = 1000)




import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import pandas as pd
import re
from scipy.ndimage import distance_transform_edt, gaussian_filter
import openai
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# Set your OpenAI API key
# Load environment variables and set OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

openai.api_key = openai_api_key


########################
# 0. MODEL DEFINITION  #
########################

# Time settings for simulation (0 to 100 simulation time mapped to years 2000-2100)
t_end = 100
dt = 0.1
time = np.arange(0, t_end, dt)
n_steps = len(time)

# Stock indices (for stock topics)
IDX_POPULATION    = 0
IDX_NATURAL_COVER = 1
IDX_BIODIVERSITY  = 2
IDX_SEA_LEVEL     = 3
IDX_RESILIENCE    = 4
IDX_INFRA         = 5
IDX_WATER         = 6
IDX_TEMPERATURE   = 7

# Labels for stock topics
stock_labels = [
    "Population",
    "Natural Cover",
    "Biodiversity",
    "Sea Level",
    "Village Resilience",
    "Infrastructure",
    "Water Resources",
    "Temperature"
]

# Adjusted Default Parameters with higher sensitivity
default_params = {
    "policy_management": 0.5,
    "climate_policy": 0.5,
    "economic_policy": 1.25,
    "land_ownership_rules": 0.5,
    "geographic_factors": 1.0,
    "environmental_standards": 0.5,
    "natural_resource_avail": 1.0,
    "ecosystem_valuation": 0.5,
    "market_forces": 1.0,
    "community_culture": 0.5,
    "technology_readiness": 0.5,
    "financial_investment": 1.0,
    "k_land_conversion": 0.001,
    "k_biodiversity_loss": 0.1,
    "k_carbon_emission": 0.2,
    "k_sea_level_rise": 2.5,
    "k_adaptation": 0.2,
    "k_flood_damage": 0.1,
    "k_infra_degradation": 0.1,
    "k_water_usage": 0.005,
    "k_population_growth": 0.03,
    "carrying_capacity": 2000,
    "k_temp": 0.1
}

def get_initial_stocks():
    stocks0 = np.zeros(8)
    stocks0[IDX_POPULATION]    = 1000
    stocks0[IDX_NATURAL_COVER] = 1000
    stocks0[IDX_BIODIVERSITY]  = 1000
    stocks0[IDX_SEA_LEVEL]     = 0
    stocks0[IDX_RESILIENCE]    = 100
    stocks0[IDX_INFRA]         = 100
    stocks0[IDX_WATER]         = 1000
    stocks0[IDX_TEMPERATURE]   = 1.2
    return stocks0

###############################
# 1. SIMULATION CORE FUNCTION #
###############################
def run_simulation(params):
    policy_management    = params["policy_management"]
    climate_policy       = params["climate_policy"]
    economic_policy      = params["economic_policy"]
    land_ownership_rules = params["land_ownership_rules"]
    geographic_factors   = params["geographic_factors"]
    environmental_standards = params["environmental_standards"]
    natural_resource_avail  = params["natural_resource_avail"]
    ecosystem_valuation     = params["ecosystem_valuation"]
    market_forces        = params["market_forces"]
    community_culture    = params["community_culture"]
    technology_readiness = params["technology_readiness"]
    financial_investment = params["financial_investment"]
    k_land_conversion    = params["k_land_conversion"]
    k_biodiversity_loss  = params["k_biodiversity_loss"]
    k_carbon_emission    = params["k_carbon_emission"]
    k_sea_level_rise     = params["k_sea_level_rise"]
    k_adaptation         = params["k_adaptation"]
    k_flood_damage       = params["k_flood_damage"]
    k_infra_degradation  = params["k_infra_degradation"]
    k_water_usage        = params["k_water_usage"]
    k_population_growth  = params["k_population_growth"]
    carrying_capacity    = params["carrying_capacity"]
    k_temp               = params["k_temp"]

    stocks = np.zeros((n_steps, 8))
    emissions = np.zeros(n_steps)
    land_conv_array = np.zeros(n_steps)
    adaptation_array = np.zeros(n_steps)
    resilience_loss_array = np.zeros(n_steps)
    net_carbon_array = np.zeros(n_steps)

    stocks[0, :] = get_initial_stocks()

    def compute_rates(stock_vals):
        pop, cover, bio, sea, resilience, infra, water, temp = stock_vals
        growth_factor = resilience / 100.0
        dPop_dt = k_population_growth * pop * growth_factor * (1 - pop / carrying_capacity)
        land_conversion = k_land_conversion * pop * market_forces * economic_policy
        land_conversion *= (1 - policy_management * land_ownership_rules * 0.1)
        dCover_dt = -land_conversion
        dBio_dt = -k_biodiversity_loss * land_conversion
        net_carbon = k_carbon_emission * land_conversion * (1 - climate_policy)
        dSea_dt = k_sea_level_rise * net_carbon * geographic_factors
        dTemp_dt = k_temp * net_carbon
        adaptation = k_adaptation * (technology_readiness + financial_investment + community_culture) / 3
        resilience_loss = k_flood_damage * sea
        dResilience_dt = adaptation - resilience_loss
        dInfra_dt = -k_infra_degradation * sea
        water_loss = k_water_usage * pop / (environmental_standards * natural_resource_avail)
        dWater_dt = -water_loss
        return (dPop_dt, dCover_dt, dBio_dt, dSea_dt, dResilience_dt, dInfra_dt, dWater_dt, dTemp_dt,
                net_carbon, land_conversion, adaptation, resilience_loss)

    for i in range(1, n_steps):
        current = stocks[i - 1, :]
        rates = compute_rates(current)
        stocks[i, IDX_POPULATION]    = current[IDX_POPULATION]    + dt * rates[0]
        stocks[i, IDX_NATURAL_COVER] = max(current[IDX_NATURAL_COVER] + dt * rates[1], 0)
        stocks[i, IDX_BIODIVERSITY]  = max(current[IDX_BIODIVERSITY]  + dt * rates[2], 0)
        stocks[i, IDX_SEA_LEVEL]     = current[IDX_SEA_LEVEL]     + dt * rates[3]
        stocks[i, IDX_RESILIENCE]    = max(current[IDX_RESILIENCE]    + dt * rates[4], 0)
        stocks[i, IDX_INFRA]         = max(current[IDX_INFRA]         + dt * rates[5], 0)
        stocks[i, IDX_WATER]         = max(current[IDX_WATER]         + dt * rates[6], 0)
        stocks[i, IDX_TEMPERATURE]   = current[IDX_TEMPERATURE]   + dt * rates[7]
        net_carbon_array[i]      = rates[8]
        land_conv_array[i]       = rates[9]
        adaptation_array[i]      = rates[10]
        resilience_loss_array[i] = rates[11]
        emissions[i] = rates[8]

    return stocks, emissions, land_conv_array, adaptation_array, resilience_loss_array, net_carbon_array

##############################
# 2. SPATIAL ANALYSIS FUNCTIONS
##############################

# Define grid size and create coordinate arrays
size = 100
x = np.linspace(-1, 1, size)
y = np.linspace(-1, 1, size)
X, Y = np.meshgrid(x, y)

# Compute radial distance and add noise to create a rough island shape
R = np.sqrt(X**2 + Y**2)
np.random.seed(42)
noise = np.random.normal(0, 0.1, (size, size))
R_noisy = R + noise
threshold = 0.7
island_mask = R_noisy < threshold

# Compute distance from the “beach” and generate an elevation map
distance_from_beach = distance_transform_edt(island_mask)
max_distance = np.max(distance_from_beach)
normalized_distance = distance_from_beach / max_distance

max_elevation = 300
exponent = 1.0
elevation_island = (normalized_distance ** exponent) * max_elevation
terrain_noise = np.random.normal(0, 10, (size, size))
baseline_elevation = np.where(island_mask, elevation_island + terrain_noise, 0)
baseline_elevation = np.clip(baseline_elevation, 0, None)

# Create a smoothed noise field for land-cover generation
random_field = np.random.rand(size, size)
smoothed_noise = gaussian_filter(random_field, sigma=5)
smoothed_noise = (smoothed_noise - smoothed_noise.min()) / (smoothed_noise.max()-smoothed_noise.min())

# Generate initial land cover classification
land_cover = np.empty((size, size), dtype=object)
for i in range(size):
    for j in range(size):
        if not island_mask[i, j]:
            land_cover[i, j] = "water"
        else:
            d = normalized_distance[i, j]
            noise_val = smoothed_noise[i, j]
            if d < 0.2:
                land_cover[i, j] = "settlement" if noise_val > 0.5 else "natural cover"
            elif d < 0.5:
                if noise_val < 0.33:
                    land_cover[i, j] = "agriculture"
                elif noise_val < 0.66:
                    land_cover[i, j] = "natural cover"
                else:
                    land_cover[i, j] = "palm oil plantation"
            else:
                if noise_val < 0.3:
                    land_cover[i, j] = "agriculture"
                elif noise_val < 0.7:
                    land_cover[i, j] = "natural cover"
                else:
                    land_cover[i, j] = "palm oil plantation"

# Map textual land cover into numerical values
land_cover_mapping = {"water": 0, "settlement": 1, "natural cover": 2, "agriculture": 3, "palm oil plantation": 4}
baseline_land_cover = np.zeros((size, size), dtype=int)
for i in range(size):
    for j in range(size):
        baseline_land_cover[i, j] = land_cover_mapping[land_cover[i, j]]

# Define infiltration factors for each land cover
infiltration_factors = {
    "water": 0.0,
    "settlement": 0.1,
    "natural cover": 1.0,
    "agriculture": 0.8,
    "palm oil plantation": 0.2
}

# Define fixed ranges for aspatial (time-series) and spatial displays
aspatial_fixed_ranges = {
    "population": (1000, 2000),
    "natural_cover": (500, 1000),
    "biodiversity": (950, 1000),
    "sea_level": (0, 50),
    "resilience": (0, 100),
    "infrastructure": (0, 150),
    "water": (0, 1000),
    "temperature": (1, 3)
}

spatial_fixed_ranges = {
    "population": (0, 3000),
    "natural_cover": (0, 1000),
    "biodiversity": (0, 1),
    "sea_level": (-100, 300),
    "resilience": (10,150 ),
    "infrastructure": (0, 1),
    "water": (100,800),
    "temperature": (1, 3)
}

##############################
# 2A. ADMINISTRATIVE BOUNDARIES
##############################
admin_boundaries = np.zeros((size, size), dtype=int)
admin_boundaries[(island_mask) & (X < -0.6)] = 1
admin_boundaries[(island_mask) & (X >= -0.6) & (X < -0.2)] = 2
admin_boundaries[(island_mask) & (X >= -0.2) & (X < 0.2)] = 3
admin_boundaries[(island_mask) & (X >= 0.2) & (X < 0.6)] = 4
admin_boundaries[(island_mask) & (X >= 0.6)] = 5

# Optional: Visualize the administrative boundaries.
plt.imshow(admin_boundaries, cmap="tab10")
plt.title("Administrative Boundaries (5 Cities)")
plt.colorbar(label="City ID")
plt.show()

# Define city-specific attribute factors.
city_attributes = {
    1: {'population_factor': 1.2, 'natural_cover_factor': 0.8, 'adaptation_factor': 1.1},
    2: {'population_factor': 0.9, 'natural_cover_factor': 1.1, 'adaptation_factor': 0.95},
    3: {'population_factor': 1.0, 'natural_cover_factor': 1.0, 'adaptation_factor': 1.0},
    4: {'population_factor': 1.3, 'natural_cover_factor': 0.7, 'adaptation_factor': 1.2},
    5: {'population_factor': 0.8, 'natural_cover_factor': 1.2, 'adaptation_factor': 0.85}
}

##############################
# 3. UPDATE SPATIAL MAP FUNCTION WITH ADMIN BOUNDARIES
##############################
def update_spatial_map_topic_time_with_admin(sim_stocks, chosen_time, chosen_topic):
    idx = min(int(float(chosen_time) / dt), n_steps - 1)
    if chosen_topic == "natural_cover":
        initial_natural = get_initial_stocks()[IDX_NATURAL_COVER]
        current_natural = sim_stocks[idx, IDX_NATURAL_COVER]
        f = current_natural / initial_natural
        new_land_cover = np.full((size, size), 3, dtype=int)
        new_land_cover[~island_mask] = 0
        mask_natural = (baseline_land_cover == land_cover_mapping["natural cover"]) & island_mask
        rng = np.random.default_rng(42)
        rand_vals = rng.random((size, size))
        new_land_cover[mask_natural & (rand_vals < f)] = 2
        new_land_cover[mask_natural & (rand_vals >= f)] = 1
        field = new_land_cover.astype(float)
    elif chosen_topic == "biodiversity":
        initial_natural = get_initial_stocks()[IDX_NATURAL_COVER]
        current_natural = sim_stocks[idx, IDX_NATURAL_COVER]
        f = current_natural / initial_natural
        new_land_cover = np.full((size, size), 3, dtype=int)
        new_land_cover[~island_mask] = 0
        mask_natural = (baseline_land_cover == land_cover_mapping["natural cover"]) & island_mask
        rng = np.random.default_rng(42)
        rand_vals = rng.random((size, size))
        new_land_cover[mask_natural & (rand_vals < f)] = 2
        new_land_cover[mask_natural & (rand_vals >= f)] = 1
        bio_field = np.zeros((size, size), dtype=float)
        bio_field[new_land_cover == 0] = 0.0
        bio_field[new_land_cover == 1] = 0.3
        bio_field[new_land_cover == 2] = 1.0
        bio_field[new_land_cover == 3] = 0.1
        bio_field_smoothed = gaussian_filter(bio_field, sigma=3)
        bio_min = np.min(bio_field_smoothed)
        bio_max = np.max(bio_field_smoothed)
        if bio_max > bio_min:
            bio_field_normalized = (bio_field_smoothed - bio_min) / (bio_max - bio_min)
        else:
            bio_field_normalized = bio_field_smoothed
        field = bio_field_normalized
    elif chosen_topic == "population":
        field = np.zeros((size, size))
        field[island_mask] = sim_stocks[idx, IDX_POPULATION]
    elif chosen_topic == "sea_level":
        remaining = baseline_elevation - sim_stocks[idx, IDX_SEA_LEVEL]
        field = np.full((size, size), -100.0)
        field[island_mask] = np.where(remaining[island_mask] < 0, -50, remaining[island_mask])
    elif chosen_topic == "temperature":
        field = np.zeros((size, size))
        field[island_mask] = sim_stocks[idx, IDX_TEMPERATURE]
    elif chosen_topic == "resilience":
        field = np.zeros((size, size))
        base_resilience = sim_stocks[idx, IDX_RESILIENCE]
        field[island_mask] = base_resilience
        for city, attrs in city_attributes.items():
            factor = attrs.get("adaptation_factor", 1.0)
            field[admin_boundaries == city] *= factor
    elif chosen_topic == "water":
        field = np.zeros((size, size))
        water_value = sim_stocks[idx, IDX_WATER]
        for lc, factor in infiltration_factors.items():
            mask = (baseline_land_cover == land_cover_mapping[lc]) & island_mask
            field[mask] = water_value * factor
    else:
        field = np.zeros((size, size))
        
    if chosen_topic == "natural_cover":
        modifier = np.ones((size, size))
        for city_id, attrs in city_attributes.items():
            modifier[admin_boundaries == city_id] = attrs.get("natural_cover_factor", 1.0)
        field = field * modifier
    if chosen_topic == "population":
        modifier = np.ones((size, size))
        for city_id, attrs in city_attributes.items():
            modifier[admin_boundaries == city_id] = attrs.get("population_factor", 1.0)
        field[island_mask] = field[island_mask] * modifier[island_mask]
    
    return field

##########################
# PRE-RUN BASELINE SIMULATION
##########################
baseline_stocks, baseline_emissions, baseline_land_conv, baseline_adaptation, baseline_res_loss, baseline_net_carbon = run_simulation(default_params)

##########################
# Legend Style Definitions
##########################
# custom_legend_style = {"textAlign": "center", "marginTop": "2px"}
# baseline_legend_box = {
#     "backgroundColor": "#000000",
#     "color": "#FFFFFF",
#     "padding": "6px 12px",
#     "marginRight": "20px",
#     "marginBottom": "20px",
#     "fontWeight": "bold",
#     "borderRadius": "3px"
# }
# scenario_legend_box = {
#     "backgroundColor": "#00ADEE",
#     "color": "#FFFFFF",
#     "padding": "6px 12px",
#     "fontWeight": "bold",
#     "marginBottom": "20px",
#     "borderRadius": "3px"
# }

##########################
# 4. DASH APP LAYOUT
##########################
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.title = "SDSS Spatial Decision Support System for Climate Action"

slider_tooltips = {
    "policy_management": "Higher values mean stronger policy management.",
    "climate_policy": "Stricter policies lower emissions and temperature rise.",
    "economic_policy": "Higher values drive aggressive growth (more emissions).",
    "land_ownership_rules": "Stronger rules protect natural cover.",
    "geographic_factors": "Higher vulnerability increases impacts.",
    "environmental_standards": "Stricter standards curb emissions.",
    "natural_resource_avail": "More resource availability supports sustainability.",
    "ecosystem_valuation": "Higher valuation encourages conservation.",
    "market_forces": "Stronger market forces increase emissions.",
    "community_culture": "Stronger culture supports sustainability.",
    "technology_readiness": "Higher readiness enables clean tech adoption.",
    "financial_investment": "More investment improves outcomes."
}
slider_params = {
    "policy_management": {"label": "Policy Management", "min": 0.0, "max": 1.0, "step": 0.05, "default": default_params["policy_management"]},
    "climate_policy": {"label": "Climate Policy", "min": 0.0, "max": 1.0, "step": 0.05, "default": default_params["climate_policy"]},
    "economic_policy": {"label": "Economic Policy", "min": 0.5, "max": 2.0, "step": 0.1, "default": default_params["economic_policy"]},
    "land_ownership_rules": {"label": "Land Ownership Rules", "min": 0.0, "max": 1.0, "step": 0.05, "default": default_params["land_ownership_rules"]},
    "geographic_factors": {"label": "Geographic Factors", "min": 0.0, "max": 2.0, "step": 0.1, "default": default_params["geographic_factors"]},
    "environmental_standards": {"label": "Environmental Standards", "min": 0.0, "max": 1.0, "step": 0.05, "default": default_params["environmental_standards"]},
    "natural_resource_avail": {"label": "Natural Resource Avail.", "min": 0.0, "max": 2.0, "step": 0.1, "default": default_params["natural_resource_avail"]},
    "ecosystem_valuation": {"label": "Ecosystem Valuation", "min": 0.0, "max": 1.0, "step": 0.05, "default": default_params["ecosystem_valuation"]},
    "market_forces": {"label": "Market Forces", "min": 0.0, "max": 2.0, "step": 0.1, "default": default_params["market_forces"]},
    "community_culture": {"label": "Community Culture", "min": 0.0, "max": 1.0, "step": 0.05, "default": default_params["community_culture"]},
    "technology_readiness": {"label": "Technology Readiness", "min": 0.0, "max": 1.0, "step": 0.05, "default": default_params["technology_readiness"]},
    "financial_investment": {"label": "Financial Investment", "min": 0.0, "max": 2.0, "step": 0.1, "default": default_params["financial_investment"]}
}
slider_effect = {
    "policy_management": "beneficial",
    "climate_policy": "beneficial",
    "economic_policy": "detrimental",
    "land_ownership_rules": "beneficial",
    "geographic_factors": "detrimental",
    "environmental_standards": "beneficial",
    "natural_resource_avail": "beneficial",
    "ecosystem_valuation": "beneficial",
    "market_forces": "detrimental",
    "community_culture": "beneficial",
    "technology_readiness": "beneficial",
    "financial_investment": "beneficial"
}
def get_slider_description(param_id, value, slider_cfg):
    default_val = slider_cfg["default"]
    effect = slider_effect.get(param_id, "beneficial")
    if abs(value - default_val) < 0.001:
        return "Status quo: conditions maintained."
    elif value < default_val:
        return "Lower value: worse outcome." if effect == "beneficial" else "Lower value: better outcome."
    else:
        return "Higher value: better outcome." if effect == "beneficial" else "Higher value: worse outcome."

def build_slider_div(param_id, cfg):
    return html.Div([
        html.Label(cfg["label"], title=slider_tooltips.get(param_id, "No description available.")),
        dcc.Slider(
            id=param_id,
            min=cfg["min"],
            max=cfg["max"],
            step=cfg["step"],
            value=cfg["default"],
            marks={str(v): str(v) for v in np.linspace(cfg["min"], cfg["max"], 5)},
            tooltip={"placement": "bottom"}
        ),
        html.Div(id=f"{param_id}-description", style={"fontSize": "0.8rem", "color": "#666", "marginTop": "5px", "fontWeight": "bold"})
    ], style={"marginBottom": "20px"})

cat1_sliders = ["policy_management", "climate_policy", "economic_policy", "land_ownership_rules"]
cat2_sliders = ["geographic_factors", "environmental_standards", "natural_resource_avail", "ecosystem_valuation"]
cat3_sliders = ["market_forces", "community_culture", "technology_readiness", "financial_investment"]

def build_category_panel(category_title, param_keys):
    return html.Div([
        html.Div(category_title, style={"backgroundColor": "#eee", "fontWeight": "bold", "padding": "6px", "textAlign": "center", "marginBottom": "10px"}),
        html.Div([build_slider_div(pk, slider_params[pk]) for pk in param_keys], style={"columnCount": 2})
    ], style={"border": "2px solid #D3D3D3", "borderRadius": "5px", "padding": "10px", "width": "29%", "margin": "10px", "display": "inline-block", "verticalAlign": "top"})

cat1_panel = build_category_panel("Governance & Institutional", cat1_sliders)
cat2_panel = build_category_panel("Environmental & Geographic", cat2_sliders)
cat3_panel = build_category_panel("Socio-Economic & Global", cat3_sliders)

topics = [
    {"label": "Population",         "value": "population"},
    {"label": "Natural Cover",        "value": "natural_cover"},
    {"label": "Biodiversity",         "value": "biodiversity"},
    {"label": "Sea Level",            "value": "sea_level"},
    {"label": "Resilience",           "value": "resilience"},
    {"label": "Water Resources",      "value": "water"},
    {"label": "Temperature",          "value": "temperature"},
    {"label": "Land Conversion",      "value": "land_conversion"},
    {"label": "Adaptation",           "value": "adaptation"},
    {"label": "Resilience Loss",      "value": "resilience_loss"},
    {"label": "Emissions",            "value": "emissions"}
]
    
topic_mapping = {
    "population":       {"type": "stock", "index": 0},
    "natural_cover":    {"type": "stock", "index": 1},
    "biodiversity":     {"type": "stock", "index": 2},
    "sea_level":        {"type": "stock", "index": 3},
    "resilience":       {"type": "stock", "index": 4},
    "infrastructure":   {"type": "stock", "index": 5},
    "water":            {"type": "stock", "index": 6},
    "temperature":      {"type": "stock", "index": 7},
    "land_conversion":  {"type": "flow",  "data": "land_conv"},
    "adaptation":       {"type": "flow",  "data": "adaptation"},
    "resilience_loss":  {"type": "flow",  "data": "resilience_loss"},
    "emissions":        {"type": "flow",  "data": "net_carbon"}
}

# Define static topic descriptions.
topic_descriptions = {
    "population": {
        "aspatial": "A time-series depicting changes in the island's population over the simulation period.",
        "spatial": "A map illustrating the distribution of population across the island."
    },
    "natural_cover": {
        "aspatial": "The time-series representing the remaining natural cover on the island.",
        "spatial": "A map showing the spatial distribution of natural cover."
    },
    "biodiversity": {
        "aspatial": "Shows changes in biodiversity as a result of land conversion.",
        "spatial": "A heatmap indicating biodiversity levels spatially."
    },
    "sea_level": {
        "aspatial": "Tracks the change of sea level over time, influenced by carbon emissions.",
        "spatial": "A map displaying sea level rise impact and inundation patterns."
    },
    "resilience": {
        "aspatial": "Represents overall village resilience from aggregated adaptation efforts.",
        "spatial": "A map showing resilience variations across cities with distinct adaptation measures."
    },
    "infrastructure": {
        "aspatial": "Time-series data on infrastructure progress or degradation over time.",
        "spatial": "Not applicable for spatial mapping."
    },
    "water": {
        "aspatial": "Represents water resource levels in the simulation over time.",
        "spatial": "A map indicating water resource availability by land cover."
    },
    "temperature": {
        "aspatial": "A time-series showing changes in average temperature over the simulation period.",
        "spatial": "A map displaying the spatial distribution of temperature."
    },
    "land_conversion": {
        "aspatial": "Shows the rate of land conversion over time.",
        "spatial": "Not directly mapped; best observed in aspatial trends."
    },
    "adaptation": {
        "aspatial": "Depicts adaptive measures and improvements over time.",
        "spatial": "Not applicable for spatial mapping."
    },
    "resilience_loss": {
        "aspatial": "Indicates losses in resilience over time due to flood and climate impacts.",
        "spatial": "Not directly mapped."
    },
    "emissions": {
        "aspatial": "Illustrates the trend of carbon emissions as a result of land conversion and policy.",
        "spatial": "Not directly mapped spatially."
    }
}

# Build description content with styles that constrain overflow.
description_content = html.Div(
    [html.Div([
         html.H4(topic.capitalize()),
         html.P("A-Spatial: " + topic_descriptions[topic]["aspatial"]),
         html.P("Spatial: " + topic_descriptions[topic]["spatial"])
     ], style={"marginBottom": "20px", "fontSize": "0.9rem", "borderBottom": "1px solid #ccc", "paddingBottom": "10px"})
     for topic in topic_descriptions.keys()
    ],
    style={"padding": "20px", "overflowY": "auto", "maxHeight": "350px", "wordWrap": "break-word", "boxSizing": "border-box"}
)

right_description_content = description_content

left_aspatial_topic_dropdown = dcc.Dropdown(
    id="left-topic-dropdown",
    options=topics,
    value="population",
    clearable=False,
    style={"width": "100%", "fontSize": "1rem", "fontWeight": "bold", "marginBottom": "10px"}
)

left_aspatial_graph = dcc.Graph(
    id="left-graph",
    style={"height": "300px", "position": "relative", "top": "-15px"}
)

# left_aspatial_legend = html.Div([
#     html.Span("BASELINE", style=baseline_legend_box),
#     html.Span("CURRENT SCENARIO", style=scenario_legend_box)
# ], style=custom_legend_style)

left_spatial_time_dropdown = dcc.Dropdown(
    id="left-spatial-time-dropdown",
    options=[{"label": f"{t} year" if t == 1 else f"{t} years", "value": str(t)} for t in [1, 5, 10, 25, 50, 100]],
    value="25",
    clearable=False,
    style={"width": "40%", "minWidth": "200px", "whiteSpace": "normal", "display": "inline-block"}
)
left_spatial_topic_dropdown = dcc.Dropdown(
    id="left-spatial-topic-dropdown",
    options=[{"label": "Population", "value": "population"},
             {"label": "Natural Cover", "value": "natural_cover"},
             {"label": "Biodiversity", "value": "biodiversity"},
             {"label": "Sea Level", "value": "sea_level"},
             {"label": "Resilience", "value": "resilience"},
             {"label": "Water Resources", "value": "water"},
             {"label": "Temperature", "value": "temperature"}],
    value="population",
    clearable=False,
    style={"width": "40%", "minWidth": "200px", "whiteSpace": "normal", "display": "inline-block"}
)
left_spatial_controls = html.Div([
    left_spatial_time_dropdown,
    left_spatial_topic_dropdown
], style={"display": "flex", "justifyContent": "center", "alignItems": "center", "gap": "10px", "marginBottom": "10px"})

left_spatial_graph = dcc.Graph(id="left-spatial-map", style={"height": "600px", "width": "600px"})

left_ask_ai_content = html.Div([
    dcc.Textarea(
        id="left-ask-ai-input",
        placeholder="How bad our temperature in 2045? or What is the change in biodiversity from 2030 to 2050?",
        style={"width": "100%", "height": "100px"}
    ),
    html.Br(),
    html.Button("Submit", id="left-ask-ai-submit"),
    html.Div(id="left-ask-ai-output", style={"marginTop": "20px", "whiteSpace": "pre-line"})
])

left_tabs = dcc.Tabs(id="left-tabs", value="aspacial", children=[
    dcc.Tab(label="A-Spatial", value="aspacial", children=[
        left_aspatial_topic_dropdown,
        left_aspatial_graph,
    #    left_aspatial_legend
    ]),
    dcc.Tab(label="Spatial", value="spatial", children=[
        left_spatial_controls,
        left_spatial_graph
    ]),
    dcc.Tab(label="Description", value="description", children=[
        description_content
    ]),
    dcc.Tab(label="Ask.AI", value="ask_ai", children=[
        left_ask_ai_content
    ])
])

right_aspatial_topic_dropdown = dcc.Dropdown(
    id="right-topic-dropdown",
    options=topics,
    value="temperature",
    clearable=False,
    style={"width": "100%", "fontSize": "1rem", "fontWeight": "bold", "marginBottom": "10px"}
)

right_aspatial_graph = dcc.Graph(
    id="right-graph",
    style={"height": "300px", "position": "relative", "top": "-15px"}
)

# right_aspatial_legend = html.Div([
#     html.Span("BASELINE", style=baseline_legend_box),
#     html.Span("CURRENT SCENARIO", style=scenario_legend_box)
# ], style=custom_legend_style)

right_spatial_time_dropdown = dcc.Dropdown(
    id="right-spatial-time-dropdown",
    options=[{"label": f"{t} year" if t == 1 else f"{t} years", "value": str(t)} for t in [1, 5, 10, 25, 50, 100]],
    value="25",
    clearable=False,
    style={"width": "40%", "minWidth": "200px", "whiteSpace": "normal", "display": "inline-block"}
)
right_spatial_topic_dropdown = dcc.Dropdown(
    id="right-spatial-topic-dropdown",
    options=[{"label": "Population", "value": "population"},
             {"label": "Natural Cover", "value": "natural_cover"},
             {"label": "Biodiversity", "value": "biodiversity"},
             {"label": "Sea Level", "value": "sea_level"},
             {"label": "Resilience", "value": "resilience"},
             {"label": "Water Resources", "value": "water"},
             {"label": "Temperature", "value": "temperature"}],
    value="population",
    clearable=False,
    style={"width": "40%", "minWidth": "200px", "whiteSpace": "normal", "display": "inline-block"}
)
right_spatial_controls = html.Div([
    right_spatial_time_dropdown,
    right_spatial_topic_dropdown
], style={"display": "flex", "justifyContent": "center", "alignItems": "center", "gap": "10px", "marginBottom": "10px"})

right_spatial_graph = dcc.Graph(id="right-spatial-map", style={"height": "600px", "width": "600px"})

right_ask_ai_content = html.Div([
    dcc.Textarea(
        id="right-ask-ai-input",
        placeholder="How bad our temperature in 2045? or What is the change in biodiversity from 2030 to 2050?",
        style={"width": "100%", "height": "100px"}
    ),
    html.Br(),
    html.Button("Submit", id="right-ask-ai-submit"),
    html.Div(id="right-ask-ai-output", style={"marginTop": "20px", "whiteSpace": "pre-line"})
])

right_tabs = dcc.Tabs(id="right-tabs", value="aspacial", children=[
    dcc.Tab(label="A-Spatial", value="aspacial", children=[
        right_aspatial_topic_dropdown,
        right_aspatial_graph,
      #  right_aspatial_legend
    ]),
    dcc.Tab(label="Spatial", value="spatial", children=[
        right_spatial_controls,
        right_spatial_graph
    ]),
    dcc.Tab(label="Description", value="description", children=[
        right_description_content
    ]),
    dcc.Tab(label="Ask.AI", value="ask_ai", children=[
        right_ask_ai_content
    ])
])

temp_c_indicator = html.Div(id="temp-c-indicator", style={"justifyContent": "center", "fontSize": "4rem", "color": "#007ACC", "fontWeight": "bold", "marginBottom": "20px", "marginTop": "20px", "height": "50px"})
temp_f_indicator = html.Div(id="temp-f-indicator", style={"justifyContent": "center", "fontSize": "2rem", "color": "#007ACC", "marginTop": "0", "height": "70px"})
temp_label = html.Div("Temperature Increase by 2100", style={"justifyContent": "center", "fontSize": "1rem", "marginTop": "10px", "fontWeight": "bold", "color": "#333", "height": "73px"})
temp_extra_info = html.Div(
    "The temperature increase is measured relative to a pre-industrial benchmark (around 14°C). "
    "This baseline represents average global temperatures before significant industrialization. "
    "Limiting the warming to 1.5°C or 2°C above this level is essential to avoid severe impacts.",
    style={"fontSize": "0.9rem", "marginTop": "10px", "marginBottom": "20px", "color": "#555"}
)

# Save button below the temperature info on right panel.
#temp_save_button = html.Button("Save Results", id="save-button",
#                style={"width": "90%", "padding": "10px", "fontSize": "0.8rem", "marginTop": "10px"})

instruction = html.Div("Adjust the sliders below to explore how different policy and environmental factors influence outcomes.",
                       style={"textAlign": "center", "marginBottom": "20px", "fontSize": "1.0rem"})

app.layout = html.Div([
    html.Div([
        html.H2("SDSS Spatial Decision Support System for Climate Action", 
                style={"textAlign": "center", "padding": "1px", "fontSize": "1.25rem"})
    ], style={"backgroundColor": "#eee", "borderRadius": "2px", "marginBottom": "20px", "border": "2px solid #D3D3D3"}),
    instruction,
    html.Div([
        html.Div(left_tabs, style={"width": "36%", "height": "435px", "display": "inline-block", "verticalAlign": "top", "padding": "10px", "border": "2px solid #D3D3D3", "borderRadius": "5px", "marginRight": "1%"}),
        html.Div(right_tabs, style={"width": "36%", "height": "435px", "display": "inline-block", "verticalAlign": "top", "padding": "10px", "border": "2px solid #D3D3D3", "borderRadius": "5px", "marginRight": "1%"}),
        html.Div([
            temp_c_indicator,
            temp_f_indicator,
            temp_label,
            temp_extra_info,
      #      temp_save_button
        ], style={"width": "20%", "height": "435px", "display": "inline-block", "verticalAlign": "top", "padding": "20px", "border": "2px solid #D3D3D3", "borderRadius": "5px", "textAlign": "center"})
    ]),
    html.Div([cat1_panel, cat2_panel, cat3_panel], style={"width": "100%", "textAlign": "center"}),
    html.Div("© GEARS (Geospatial Enabler Analytics Resources System and Platform) - Telkom Indonesia", style={"textAlign": "center", "fontSize": "12px", "color": "gray"})
], style={"margin": "20px"})

##########################
# 5. A-SPATIAL FIGURE FUNCTION
##########################
def build_topic_figure(topic_key, sim_stocks, sim_land_conv, sim_adaptation, sim_res_loss, sim_net_carbon):
    new_baseline_color = "#000000"
    new_scenario_color = "#00ADEE"
    scenario_line_width = 6
    baseline_line_width = 2

    info = topic_mapping[topic_key]
    if info["type"] == "stock":
        idx = info["index"]
        model_data = sim_stocks[:, idx]
        baseline_data = baseline_stocks[:, idx]
        yaxis_label = stock_labels[idx]
    else:
        data_key = info["data"]
        if data_key == "land_conv":
            model_data = sim_land_conv
            baseline_data = baseline_land_conv
            yaxis_label = "Land Conversion"
        elif data_key == "adaptation":
            model_data = sim_adaptation
            baseline_data = baseline_adaptation
            yaxis_label = "Adaptation"
        elif data_key == "resilience_loss":
            model_data = sim_res_loss
            baseline_data = baseline_res_loss
            yaxis_label = "Resilience Loss"
        elif data_key == "net_carbon":
            model_data = sim_net_carbon
            baseline_data = baseline_net_carbon
            yaxis_label = "Emissions"
        else:
            model_data = np.zeros(n_steps)
            baseline_data = np.zeros(n_steps)
            yaxis_label = ""
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time, y=model_data,
        mode="lines",
        name="Current Scenario",
        line=dict(color=new_scenario_color, width=scenario_line_width, dash="solid"),
        hovertemplate="Year: %{x:.1f}<br>Value: %{y:.1f}"
    ))
    fig.add_trace(go.Scatter(
        x=time, y=baseline_data,
        mode="lines",
        name="Baseline",
        line=dict(color=new_baseline_color, width=baseline_line_width, dash="solid"),
        hovertemplate="Year: %{x:.1f}<br>Value: %{y:.1f}"
    ))
    xaxis_dict = dict(
        range=[0, 100],
        tickvals=[0, 20, 40, 60, 80, 100],
        ticktext=[str(2000 + t) for t in [0, 20, 40, 60, 80, 100]],
        title="Year",
        showgrid=True,
        gridcolor="#E0E0E0"
    )
    yaxis_dict = dict(
        title=yaxis_label,
        showgrid=True,
        gridcolor="#E0E0E0"
    )
    if topic_key in aspatial_fixed_ranges:
        yaxis_dict["range"] = aspatial_fixed_ranges[topic_key]
    fig.update_layout(
        xaxis=xaxis_dict,
        yaxis=yaxis_dict,
        margin=dict(l=70, r=20, t=50, b=50),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False
    )
    return fig

##########################
# 6. CALLBACKS
##########################
sea_level_colorscale = [
    [0.0, "blue"],
    [0.125, "red"],
    [0.25, "yellow"],
    [1.0, "green"]
]

latest_left_map = None
latest_right_map = None
latest_simulation_df = None
latest_simulation_stocks = None

@app.callback(
    Output("left-graph", "figure"),
    Output("right-graph", "figure"),
    Output("left-spatial-map", "figure"),
    Output("right-spatial-map", "figure"),
    Output("temp-c-indicator", "children"),
    Output("temp-f-indicator", "children"),
    Output("policy_management-description", "children"),
    Output("climate_policy-description", "children"),
    Output("economic_policy-description", "children"),
    Output("land_ownership_rules-description", "children"),
    Output("geographic_factors-description", "children"),
    Output("environmental_standards-description", "children"),
    Output("natural_resource_avail-description", "children"),
    Output("ecosystem_valuation-description", "children"),
    Output("market_forces-description", "children"),
    Output("community_culture-description", "children"),
    Output("technology_readiness-description", "children"),
    Output("financial_investment-description", "children"),
    [Input(pid, "value") for pid in slider_params] +
    [Input("left-topic-dropdown", "value"),
     Input("right-topic-dropdown", "value"),
     Input("left-spatial-time-dropdown", "value"),
     Input("left-spatial-topic-dropdown", "value"),
     Input("right-spatial-time-dropdown", "value"),
     Input("right-spatial-topic-dropdown", "value")]
)
def update_dashboard(*inputs):
    num_sliders = len(slider_params)
    slider_values = inputs[:num_sliders]
    left_topic = inputs[num_sliders]
    right_topic = inputs[num_sliders + 1]
    left_spatial_time = inputs[num_sliders + 2]
    left_spatial_topic = inputs[num_sliders + 3]
    right_spatial_time = inputs[num_sliders + 4]
    right_spatial_topic = inputs[num_sliders + 5]

    updated_params = dict(default_params)
    for i, pid in enumerate(slider_params.keys()):
        updated_params[pid] = slider_values[i]

    sim_stocks, sim_emissions, sim_land_conv, sim_adaptation, sim_res_loss, sim_net_carbon = run_simulation(updated_params)
    
    global latest_simulation_stocks
    latest_simulation_stocks = sim_stocks

    left_fig = build_topic_figure(left_topic, sim_stocks, sim_land_conv, sim_adaptation, sim_res_loss, sim_net_carbon)
    right_fig = build_topic_figure(right_topic, sim_stocks, sim_land_conv, sim_adaptation, sim_res_loss, sim_net_carbon)

    left_field = update_spatial_map_topic_time_with_admin(sim_stocks, left_spatial_time, left_spatial_topic)
    if left_spatial_topic == "natural_cover":
        left_colorscale = [
            [0.0, "blue"],
            [0.25, "blue"],
            [0.25, "red"],
            [0.5, "red"],
            [0.5, "green"],
            [0.75, "green"],
            [0.75, "brown"],
            [1.0, "brown"]
        ]
        left_zmin, left_zmax = 0, 3
    elif left_spatial_topic == "biodiversity":
        left_colorscale = "viridis"
        left_zmin, left_zmax = 0, 1
    else:
        fixed_range_left = spatial_fixed_ranges.get(left_spatial_topic, (None, None))
        left_colorscale = sea_level_colorscale if left_spatial_topic == "sea_level" else "viridis"
        left_zmin, left_zmax = fixed_range_left

    left_spatial_fig = go.Figure(data=go.Heatmap(
        z=left_field,
        colorscale=left_colorscale,
        zmin=left_zmin,
        zmax=left_zmax
    ))
    left_spatial_fig.update_layout(
        title=f"Spatial Distribution: {left_spatial_topic.capitalize()} at {left_spatial_time} year",
        width=510, height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    right_field = update_spatial_map_topic_time_with_admin(sim_stocks, right_spatial_time, right_spatial_topic)
    if right_spatial_topic == "natural_cover":
        right_colorscale = [
            [0.0, "blue"],
            [0.25, "blue"],
            [0.25, "red"],
            [0.5, "red"],
            [0.5, "green"],
            [0.75, "green"],
            [0.75, "brown"],
            [1.0, "brown"]
        ]
        right_zmin, right_zmax = 0, 3
    elif right_spatial_topic == "biodiversity":
        right_colorscale = "viridis"
        right_zmin, right_zmax = 0, 1
    else:
        fixed_range_right = spatial_fixed_ranges.get(right_spatial_topic, (None, None))
        right_colorscale = sea_level_colorscale if right_spatial_topic == "sea_level" else "viridis"
        right_zmin, right_zmax = fixed_range_right

    right_spatial_fig = go.Figure(data=go.Heatmap(
        z=right_field,
        colorscale=right_colorscale,
        zmin=right_zmin,
        zmax=right_zmax
    ))
    right_spatial_fig.update_layout(
        title=f"Spatial Distribution: {right_spatial_topic.capitalize()} at {right_spatial_time} year",
        width=510, height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    final_temp_c = sim_stocks[-1, IDX_TEMPERATURE]
    temp_string_c = f"{final_temp_c:+.1f}°C"
    final_temp_f = final_temp_c * 1.8
    temp_string_f = f"{final_temp_f:+.1f}°F"
    
    slider_descs = [get_slider_description(pid, slider_values[i], slider_params[pid]) for i, pid in enumerate(slider_params.keys())]

    global latest_simulation_df, latest_left_map, latest_right_map
    chosen_time_csv = str(t_end)
    idx_csv = n_steps - 1
    pop_field_csv       = update_spatial_map_topic_time_with_admin(sim_stocks, chosen_time_csv, "population")
    natcov_field_csv    = update_spatial_map_topic_time_with_admin(sim_stocks, chosen_time_csv, "natural_cover")
    biodiv_field_csv    = update_spatial_map_topic_time_with_admin(sim_stocks, chosen_time_csv, "biodiversity")
    sea_level_field_csv = update_spatial_map_topic_time_with_admin(sim_stocks, chosen_time_csv, "sea_level")
    resilience_field_csv= update_spatial_map_topic_time_with_admin(sim_stocks, chosen_time_csv, "resilience")
    water_field_csv     = update_spatial_map_topic_time_with_admin(sim_stocks, chosen_time_csv, "water")
    temp_field_csv      = update_spatial_map_topic_time_with_admin(sim_stocks, chosen_time_csv, "temperature")
    
    land_conv_value = sim_land_conv[idx_csv]
    adaptation_value = sim_adaptation[idx_csv]
    res_loss_value = sim_res_loss[idx_csv]
    emissions_value = sim_net_carbon[idx_csv]
    
    land_conv_field   = np.full((size, size), land_conv_value)
    adaptation_field  = np.full((size, size), adaptation_value)
    res_loss_field    = np.full((size, size), res_loss_value)
    emissions_field   = np.full((size, size), emissions_value)
    
    grid_ids = np.arange(size * size)
    Y_flat = Y.flatten()
    X_flat = X.flatten()
    elevation_flat = baseline_elevation.flatten()
    inundation_flat = sea_level_field_csv.flatten()
    infra_flat = np.zeros_like(inundation_flat)
    pop_flat = pop_field_csv.flatten()
    natcov_flat = natcov_field_csv.flatten()
    biodiv_flat = biodiv_field_csv.flatten()
    resilience_flat = resilience_field_csv.flatten()
    water_flat = water_field_csv.flatten()
    temp_flat = temp_field_csv.flatten()
    
    land_conv_flat = land_conv_field.flatten()
    adaptation_flat = adaptation_field.flatten()
    res_loss_flat = res_loss_field.flatten()
    emissions_flat = emissions_field.flatten()
    
    latest_simulation_df = pd.DataFrame({
        "grid_id": grid_ids,
        "y": Y_flat,
        "x": X_flat,
        "elevation": elevation_flat,
        "inundation": inundation_flat,
        "population": pop_flat,
        "natural_cover": natcov_flat,
        "biodiversity": biodiv_flat,
        "sea_level": sea_level_field_csv.flatten(),
        "resilience": resilience_flat,
        "water": water_flat,
        "temperature": temp_flat,
        "land_conversion": land_conv_flat,
        "adaptation": adaptation_flat,
        "resilience_loss": res_loss_flat,
        "emissions": emissions_flat
    })
    
    latest_left_map = left_spatial_fig
    latest_right_map = right_spatial_fig

    return (
        left_fig,
        right_fig,
        left_spatial_fig,
        right_spatial_fig,
        temp_string_c,
        temp_string_f,
        *slider_descs
    )

def save_map_images_and_simulation():
    try:
        # Save spatial maps as images
        pio.write_image(latest_left_map, "left_spatial_map.png")
        pio.write_image(latest_right_map, "right_spatial_map.png")
        
        # Create a DataFrame for aspatial (time-series) results
        # Add a "Year" column by converting simulation time to calendar years (2000 + time)
        if latest_simulation_stocks is not None:
            df_aspatial = pd.DataFrame(latest_simulation_stocks, columns=stock_labels)
            df_aspatial.insert(0, "Year", 2000 + time)
        else:
            df_aspatial = pd.DataFrame()
        
        # Save both spatial and aspatial results to separate sheets of one Excel file
        with pd.ExcelWriter("simulation_results.xlsx") as writer:
            latest_simulation_df.to_excel(writer, sheet_name="Spatial", index=False)
            df_aspatial.to_excel(writer, sheet_name="A-Spatial", index=False)
        return True
    except Exception as e:
        print("Error saving images and XLS:", e)
        return False

# @app.callback(
#     Output("save-button", "style"),
#     Input("save-button", "n_clicks")
# )
# def update_save_button(n_clicks):
#     base_style = {"width": "90%", "padding": "10px", "fontSize": "0.8rem", "marginTop": "5px"}
#     if n_clicks is None or n_clicks == 0:
#         return base_style
#     success = save_map_images_and_simulation()
#     style = base_style.copy()
#     if success:
#         style.update({"backgroundColor": "green", "color": "white"})
#     else:
#         style.update({"backgroundColor": "red", "color": "white"})
#     return style

##########################
# 7. ASK.AI CALLBACKS
##########################
# Define a mapping of topic keywords to stock indices for range queries.
topic_keywords = {
    "temperature": IDX_TEMPERATURE,
    "population": IDX_POPULATION,
    "natural": IDX_NATURAL_COVER,
    "biodiversity": IDX_BIODIVERSITY,
    "sea": IDX_SEA_LEVEL,
    "resilience": IDX_RESILIENCE,
    "infrastructure": IDX_INFRA,
    "water": IDX_WATER
}

@app.callback(
    Output("left-ask-ai-output", "children"),
    Input("left-ask-ai-submit", "n_clicks"),
    State("left-ask-ai-input", "value")
)
def left_ask_ai(n_clicks, command):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not command:
        return "Please enter a query."
    
    command_lower = command.lower()
    years = re.findall(r"(20\d{2}|19\d{2})", command)
    global latest_simulation_stocks

    # If two or more years are mentioned, try to identify the topic and compute a range difference.
    if len(years) >= 2:
        topic_found = None
        topic_idx = None
        for keyword, idx in topic_keywords.items():
            if keyword in command_lower:
                topic_found = keyword
                topic_idx = idx
                break
        if topic_found is not None:
            year1 = int(years[0])
            year2 = int(years[1])
            if year1 < 2000 or year1 > 2100 or year2 < 2000 or year2 > 2100:
                return f"One or both years are out of simulation range (2000-2100)."
            sim_time1 = year1 - 2000
            sim_time2 = year2 - 2000
            index_time1 = int(sim_time1 / dt)
            index_time2 = int(sim_time2 / dt)
            if latest_simulation_stocks is not None:
                val1 = latest_simulation_stocks[index_time1, topic_idx]
                val2 = latest_simulation_stocks[index_time2, topic_idx]
                change = val2 - val1
                return f"The change in {topic_found} from {year1} to {year2} is {change:+.1f}."
    
    # If a single year is mentioned, return the value at that year.
    m = re.search(r"(20\d{2}|19\d{2})", command)
    if m:
        year = int(m.group(0))
        if year < 2000 or year > 2100:
            return f"Year {year} is out of simulation range (2000-2100)."
        sim_time = year - 2000
        index_time = int(sim_time / dt)
        topic_found = None
        topic_idx = None
        for keyword, idx in topic_keywords.items():
            if keyword in command_lower:
                topic_found = keyword
                topic_idx = idx
                break
        if topic_found and latest_simulation_stocks is not None:
            if index_time >= latest_simulation_stocks.shape[0]:
                index_time = latest_simulation_stocks.shape[0] - 1
            value = latest_simulation_stocks[index_time, topic_idx]
            answer = f"In {year}, the estimated {topic_found} is {value:.1f}."
            if topic_found == "temperature":
                answer += " The temperature change appears moderate."
            return answer
    try:
        prompt = f"Based on the climate action simulation results, answer the following query: '{command}'. Provide a concise answer."
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=200
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"

@app.callback(
    Output("right-ask-ai-output", "children"),
    Input("right-ask-ai-submit", "n_clicks"),
    State("right-ask-ai-input", "value")
)
def right_ask_ai(n_clicks, command):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not command:
        return "Please enter a query."
    
    command_lower = command.lower()
    years = re.findall(r"(20\d{2}|19\d{2})", command)
    global latest_simulation_stocks

    # If two or more years are mentioned, try to identify the topic and compute a range difference.
    if len(years) >= 2:
        topic_found = None
        topic_idx = None
        for keyword, idx in topic_keywords.items():
            if keyword in command_lower:
                topic_found = keyword
                topic_idx = idx
                break
        if topic_found is not None:
            year1 = int(years[0])
            year2 = int(years[1])
            if year1 < 2000 or year1 > 2100 or year2 < 2000 or year2 > 2100:
                return f"One or both years are out of simulation range (2000-2100)."
            sim_time1 = year1 - 2000
            sim_time2 = year2 - 2000
            index_time1 = int(sim_time1 / dt)
            index_time2 = int(sim_time2 / dt)
            if latest_simulation_stocks is not None:
                val1 = latest_simulation_stocks[index_time1, topic_idx]
                val2 = latest_simulation_stocks[index_time2, topic_idx]
                change = val2 - val1
                return f"The change in {topic_found} from {year1} to {year2} is {change:+.1f}."
    
    # If a single year is mentioned, return the value at that year.
    m = re.search(r"(20\d{2}|19\d{2})", command)
    if m:
        year = int(m.group(0))
        if year < 2000 or year > 2100:
            return f"Year {year} is out of simulation range (2000-2100)."
        sim_time = year - 2000
        index_time = int(sim_time / dt)
        topic_found = None
        topic_idx = None
        for keyword, idx in topic_keywords.items():
            if keyword in command_lower:
                topic_found = keyword
                topic_idx = idx
                break
        if topic_found and latest_simulation_stocks is not None:
            if index_time >= latest_simulation_stocks.shape[0]:
                index_time = latest_simulation_stocks.shape[0] - 1
            value = latest_simulation_stocks[index_time, topic_idx]
            answer = f"In {year}, the estimated {topic_found} is {value:.1f}."
            if topic_found == "temperature":
                answer += " The temperature rise appears moderate."
            return answer
    try:
        prompt = f"Based on the climate action simulation results, answer the following query: '{command}'. Provide a concise answer."
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=200
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"

##########################
# 8. RUN SERVER
##########################
if __name__ == "__main__":
    app.run_server(debug=True)

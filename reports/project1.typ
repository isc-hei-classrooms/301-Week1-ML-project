#import "@preview/callisto:0.2.4"
#import "@preview/lilaq:0.5.0" as lq
#import "@preview/touying:0.6.1": *
#import themes.simple: *

#set text(
  font: "Source Sans 3"
)

#show: simple-theme.with(
  aspect-ratio: "16-9"
)

#let TODO = box(width: 1fr, fill: yellow, inset: .4em)[TODO]
#let notebook = json("/notebooks/project_1.ipynb")

#let (render, Cell, In, Out, output) = callisto.config(
   nb: notebook,
)

#let cols = (
  surface: "surf_hab",
  mat_quality: "qualite_materiau",
  surface_below: "surface_sous_sol",
  glob_quality: "qualite_globale",
  car_spaces: "n_garage_voitures",
  toilets: "n_toilettes",
  fireplaces: "n_cheminees",
  rooms: "n_pieces",
  kitchens: "n_cuisines",
  bedrooms: "n_chambres_coucher",
  year: "annee_vente",
  price: "prix",
  roof_type: "type_toit",
  house_type: "type_batiment",
  kitchen_quality: "qualite_cuisine",
  surface_garden: "surface_jardin"
)
#let numerical = (
  cols.surface, cols.mat_quality, cols.surface_below,
  cols.glob_quality, cols.car_spaces, cols.toilets,
  cols.fireplaces, cols.rooms, cols.kitchens,
  cols.bedrooms, cols.year, cols.price,
  cols.kitchen_quality, cols.surface_garden
)

#let raw-data = csv("/data/raw/maisons.csv", row-type: dictionary)
#let data = (:)
#for col in cols.values() {
  let values = raw-data.map(r => r.at(col))
  if col == cols.kitchen_quality {
    values = values.map(v => (
      mediocre: 0,
      moyenne: 1,
      bonne: 2,
      excellente: 3
    ).at(v))
  } else if col in numerical {
    values = values.map(float)
  }
  data.insert(col, values)
}


== Business Question

What is the value of a house, given its different characteristics ?

#figure(
  image(
    output(14).source,
    height: 9.5cm
  )
)

== Dataset Presentation

Preliminary analysis and possible models

Public data on house sales in a USA county

Gives an idea of what the housing market looks like

1364 entrypoints, each with 15 characteristics

== Data Description

#let num(body) = [(#text(fill: blue)[N]) #body]
#let cat(body) = [(#text(fill: red)[C]) #body]

#columns(2)[
- #num[Surface]
- #num[Materials quality]
- #num[Underground surface]
- #num[Global quality]
- #num[\# of parking spaces]
- #num[\# of toilets]
- #num[\# of fireplaces]
- #num[\# of rooms]
- #num[\# of kitchens]
- #num[\# of bedrooms]
- #num[Sale year]
- #num[Price]
- #cat[Type of roof]
- #cat[Type of house]
- #num[Kitchen quality]
- #num[Garden surface]
#v(1fr)
#num[] = Numerical\
#cat[] = Categorical
]

/*
#table(
  columns: 2,
  table.header[Column header][Description],
  [surf_hab], [surface habitable totale (sans-sous sol), en pieds carrés],
  [qualite_materiau], [qualité du matériau de la maison, échelle de 1 à 10],
  [surface_sous_sol], [surface totale du sous-sol en pieds-carrés],
  [qualite_globale], [qualité globale de la maison, échelle de 1 à 10],
  [n_garage_voitures], [capacité du garage en nombre de voitures],
  [n_toilettes], [nombre de toilettes],
  [n_cheminees], [nombre de cheminées],
  [n_pieces], [nombre de pièces (sans compter les salles de bain)],
  [n_cuisines], [nombre de cuisines],
  [n_chambres_coucher], [nombre de cuisines à coucher],
  [annee_vente], [année de vente de la maison],
  [prix], [prix de vente, en USD],
  [type_toit], [type de toit (1 pans, 2 pans, 4 pans, mansarde, plat)],
  [type_batiment], [type de bâtiment (maison individuelle, maison individuelle reconvertie en duplex, duplex, maison en millieu de rangée, maison en fin de rangée)],
  [qualite_cuisine], [qualité de la cuisine (excellente, bonne, moyenne, médiocre)],
  [surface_jardin], [surface du jardin, en pieds carrés]
)
*/

== Exploratory Data Analysis

#stack(dir: ltr, spacing: 1em)[
  #figure(
    image(output(8).source, height: 75%),
    caption: [Correlation between variables]
  )
][
  *High correlations with price*
  - materials quality (0.77)
  - surface (0.76)
  - underground surface (0.68)
  - number of parking spaces (0.65)
  - number of rooms (0.59)
  #v(1fr)
  Correlation between garden\ surface and \# of toilets: 0.93!

]

/*
---

Prices seem to be correlated with (>0.5):
- materials quality (0.77)
- surface (0.76)
- underground surface (0.68)
- number of parking spaces (0.65)
- number of rooms (0.59)

However, some of these variable seem to be inter-correlated:
- number of rooms / surface (0.79)
- surface / underground surface (0.61)

Fun fact, the number of toilets and garden surface have a very high correlation (0.93 !)

#lq.diagram(
  width: 20cm,
  height: 7cm,
  ylabel: [Number of toilets],
  xlabel: [Garden surface],
  lq.scatter(
    data.at(cols.surface_garden),
    data.at(cols.toilets),
    mark: "x",
    size: 8pt
  )
)
*/

/*
---

#let price-n = 12
#let price-bins = (0,) * price-n
#let prices = data.at(cols.price)
#let price-min = calc.min(..prices)
#let price-max = calc.max(..prices)
#let price-bin-w = (price-max - price-min) / (price-n - 1)
#for p in data.at(cols.price) {
  let i = calc.floor((p - price-min) / price-bin-w)
  price-bins.at(i) += 1
}

#lq.diagram(
  width: 20cm,
  height: 10cm,
  xlabel: [Price (MCHF)],
  ylabel: [Number of sales],
  lq.bar(
    range(price-n).map(i => (price-min + (i + 0.5) * price-bin-w) / 1e6),
    price-bins,
    width: price-bin-w / 1e6
  )
)
*/

== Modelling

Polynomial model of degree 6 based on multiple variables:
- Surface
- Materials quality
- \# of rooms
- \# of parking spaces
- "Is premium" (whether global quality > 6/10)


== Model Performance and Validatation

#table(
  columns: (auto, 1fr, 1fr),
  inset: (x: .4em, y: .4em),
  table.header[][*Linear Model*][*Polynomial Model*],
  [*Training Score*#footnote[Mean absolute error]<fn-mae>], [CHF 37'861.152], [CHF 26'027.110],
  [*Validation Score*@fn-mae], [CHF 37'032.211], [CHF 24'633.376]
)


== Business Conclusions and Next Steps

#TODO

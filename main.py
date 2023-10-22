import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import pydeck as pdk
import webbrowser
import seaborn as sns
import matplotlib.pyplot as plt

def text_to_html_list(text):
    segments = text.split("//")
    html_list = ""
    for segment in segments:
        html_list += f"<br>{segment}"
    return html_list

def format_value(x):
    return f"{x:.3f}"

def reset():
    st.session_state.selection = 'Aucun'




link ="https://www.data.gouv.fr/fr/datasets/r/64e02cff-9e53-4cb2-adfd-5fcc88b2dc09"
@st.cache_data
def load_clean_data(url):
    date_limite = datetime.now() - timedelta(days=3 * 30)

    fuel_color_mapping = {
        "E10": [51, 255, 67],
        "SP95": [15, 193, 46],
        "Gazole": [253, 185, 47],
        "SP98": [65, 183, 42],
        "GPLc": [41, 211, 193],
        "E85": [51, 255, 67]

    }

    column_data_types = {
        'id': str,
        'cp': str,
        'pop': str,
        'adresse': str,
        'ville': str,
        'horaires': str,
        'geom': str,
        'prix_maj': str,
        'prix_id': float,
        'prix_valeur': float,
        'prix_nom': str,
        'com_arm_code': str,
        'com_arm_name': str,
        'epci_code': str,
        'epciname': str,
        'dep_code': str,
        'dep_name': str,
        'reg_code': str,
        'reg_name': str,
        'com_code': str,
        'com_name': str,
        'services_service': str,
        'horaires_automate_24_24': str
    }
    data = pd.read_csv(link, delimiter=';', encoding='utf-8',dtype=column_data_types , na_values=['nan'] )
    data["Essence"] = data["prix_nom"]
    data["link_station"] = "https://www.google.fr/maps/place/" + data["geom"]
    data['services_service'] = data['services_service'].fillna("Non connu")
    data['services_service'] = data['services_service'].apply(text_to_html_list)
    data['prix_maj'] = data['prix_maj'].str[:10]
    data['Région'] = data['reg_name']
    data['Ville'] = data['com_name']
    data['Département'] = data['dep_name']
    colum_delete = ["dep_name","com_name","reg_name","prix_nom","epci_name","epci_code","com_arm_code","com_arm_name","pop","adresse","ville","com_code","cp","dep_code","reg_code"]
    for i in colum_delete :
        del data[i]

    purge_column = ["prix_maj","prix_id","prix_valeur","Essence","horaires_automate_24_24",'Ville','Département','Région']
    for a in purge_column :
        data = data.dropna(subset=[a])

    data['prix_maj'] = data['prix_maj'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    data = data[data['prix_maj'] >= date_limite]

    data[['latitude', 'longitude']] = data['geom'].str.split(',', expand=True)
    data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce', downcast='float')
    data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce', downcast='float')
    data['horaires_automate_24_24'] = np.where(data['horaires_automate_24_24'] == "Oui", True, False)
    date_la_plus_recente = data['prix_maj'].max()
    Ville = sorted(list(data['Ville'].unique()))
    Dep = sorted(list(data['Département'].unique()))
    Region = sorted(list(data['Région'].unique()))
    Essence_liste = sorted(list(data['Essence'].unique()))
    Ville.insert(0, "Aucun")
    Dep.insert(0, "Aucun")
    Region.insert(0, "Aucun")
    data = data.drop_duplicates(subset=["geom", "Essence"], keep="first")
    data["fill_color"] = data["Essence"].apply(lambda x: fuel_color_mapping.get(x, [255, 255, 255]))

    return data , date_la_plus_recente , Ville ,Dep ,Region,Essence_liste

def main(data, date_la_plus_recente, Ville, Dep, Region,Essence_liste):
    st.title("Le prix de l'essence en France :fuelpump:")
    st.sidebar.header("Le prix de l'essence en France :fuelpump:")
    st.sidebar.write("**Bienvenue sur cette page d'aide à la recherche des bonnes affaires en essence !**")
    st.sidebar.write("Cette page utilise des données libre de droit actualisé tous les jours par le gouvernement.")
    st.sidebar.write("Derniére Actualisation des données : "+str(date_la_plus_recente.strftime('%d/%m/%Y')))
    st.sidebar.write("Voici le lien : https://www.data.gouv.fr/fr/datasets/prix-des-carburants-en-france-flux-instantane/")
    st.sidebar.write("*PS: une V2 de ce dataset est en cours de construction*")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("Fait par Axel AUTOGUE, le 23/10/2023")
    st.sidebar.write("LinkedIn : www.linkedin.com/in/axel-autogue-5511971a2")
    st.sidebar.write("GitHub : https://github.com/LeXA7513")


    selected_reg = st.selectbox("Sélectionnez une Région :", Region,on_change=reset)
    selected_dep = st.selectbox("Sélectionnez un Département :", Dep,on_change=reset)
    selected_city = st.selectbox("Sélectionnez une Ville :", Ville,on_change=reset)
    st.write("*PS : La zone géographique la plus grande est prioritaire dans la recherche*")

    tooltip = {
        "html": "<b>{Essence}</b> à <b>{prix_valeur}</b> Eur par Litre <br>Service de la Station :  {services_service}",
        "style": {"background": "grey", "color": "white", "font-family": '"Helvetica Neue", Arial', "z-index": "10000"},
    }

    radius_taille = 500
    elevation_scale_taille = 10000



    if selected_reg != "Aucun" :
       filtered_data = data[data['Région'] == selected_reg]
    elif selected_dep != "Aucun" :
       filtered_data = data[data['Département'] == selected_dep]
       radius_taille = 250
    elif selected_city != "Aucun" :
       filtered_data = data[data['Ville'] == selected_city]
       elevation_scale_taille = 1000
       radius_taille = 100
    else :
        filtered_data = data

    view = pdk.data_utils.compute_view(filtered_data[["longitude", "latitude"]])
    view.pitch = 15


    column_layer = pdk.Layer(
        "ColumnLayer",
        data=filtered_data,
        get_position=["longitude", "latitude"],
        get_elevation="prix_valeur",
        elevation_scale=elevation_scale_taille,
        radius=radius_taille,
        get_fill_color="fill_color",
        pickable=True,
        auto_highlight=True,
    )
    st.pydeck_chart(pdk.Deck(
            column_layer,
            initial_view_state=view,
            tooltip=tooltip,
            map_provider="mapbox",
            map_style=pdk.map_styles.ROAD,
            )
        )



    essence_groups = filtered_data.groupby('Essence')
    essence_stats = essence_groups['prix_valeur'].agg(['min', 'mean', 'max'])
    essence_stats = essence_stats.rename(columns={"min":"Min","mean":"Mean","max":"Max"})

    min_indices = filtered_data.groupby('Essence')['prix_valeur'].idxmin()
    essence_link = filtered_data.loc[min_indices, ['link_station']]
    essence_link = essence_link.set_index(filtered_data.loc[min_indices, 'Essence'])

    essence_stats["Min"] = essence_stats["Min"].apply(format_value)
    essence_stats["Mean"] = essence_stats["Mean"].apply(format_value)
    essence_stats["Max"] = essence_stats["Max"].apply(format_value)


    col1, col2 = st.columns([0.7,0.3])
    col1.write("Tableau des prix à la pompe (Eur/L) :")
    with col1:
        st.dataframe(essence_stats)
    col2.write("Liens vers la pompe à essence la moins chére par carburant :")
    with col2:
        for essence, row in essence_link.iterrows():
            button_label = str(essence)
            if st.button(button_label) :
                webbrowser.open_new_tab(row["link_station"])


    liste_geo = list(["Région","Département","Ville"])
    selected_type = st.selectbox("Sélectionnez une zone Géographique :", liste_geo ,on_change=reset)
    selected_essence = st.selectbox("Sélectionnez une essence :", Essence_liste ,on_change=reset)

    if selected_type == "Région" :
        options = st.multiselect(
            'Sélectionnez une ou des '+str(selected_type)+"(s) :",
            Region[1:],
            ["Île-de-France"])
        if len(options) == 0:
            data_filtre = data[data['Essence'] == selected_essence]
        else:
            data_filtre = data.loc[(data['Essence'] == selected_essence) & (data['Région'].isin(options))]

    elif selected_type == "Département":
        options = st.multiselect(
            'Sélectionnez une ou des ' + str(selected_type) + "(s) :",
            Dep[1:],
            ["Paris"])
        if len(options) == 0:
            data_filtre = data[data['Essence'] == selected_essence]
        else:
            data_filtre = data.loc[(data['Essence'] == selected_essence) & (data['Département'].isin(options))]
    else:
        options = st.multiselect(
            'Sélectionnez une ou des ' + str(selected_type) + "(s) :",
            Ville[1:],
            ["Paris"])
        if len(options) == 0:
            data_filtre = data[data['Essence'] == selected_essence]
        else:
            data_filtre = data.loc[(data['Essence'] == selected_essence) & (data['Ville'].isin(options))]

    g = sns.catplot(x='Essence', y='prix_valeur', hue=selected_type, kind='box', data=data_filtre, legend_out=True)
    g.fig.suptitle("Prix de l'essence par zone géographique")
    g.set_axis_labels("Type d'Essence", "Prix en Eur/L")
    g.set_xticklabels(rotation=45)
    plt.tight_layout()
    st.pyplot(g)

    prices_by_region = data_filtre.groupby([selected_type, 'Essence'])['prix_valeur'].agg(['min', 'mean', 'max','count']).unstack()
    prices_by_region.columns = [f'{col[1]}_{col[0]}' for col in prices_by_region.columns]
    for col in prices_by_region.columns:
        if col != 'count':
            prices_by_region[col] = prices_by_region[col].map(format_value)

    st.dataframe(prices_by_region.style.highlight_min(axis=0))


    selected_type_bar_plot = st.selectbox("Sélectionne une zone Géographique :", liste_geo, on_change=reset)

    if selected_type_bar_plot == "Région":
        options_bar_plot = st.multiselect(
            'Sélectionne une ou des ' + str(selected_type_bar_plot) + "(s) :",
            Region[1:],
            ["Île-de-France"])
        if len(options_bar_plot) != 0:
            data_filtre_bar_plot = data[data['Région'].isin(options_bar_plot)]
            figure_barplot = plt.figure(figsize=(8, 6))
            sns.barplot(x=selected_type_bar_plot, y="prix_valeur", hue="Essence", data=data_filtre_bar_plot)
            plt.xlabel(str(selected_type_bar_plot))
            plt.ylabel("Eur/L")
            plt.title("Prix de l'essence moyen à la pompe")
            plt.xticks(rotation=45)
            st.pyplot(figure_barplot)

    elif selected_type_bar_plot == "Département":
        options_bar_plot = st.multiselect(
            'Sélectionne une ou des ' + str(selected_type_bar_plot) + "(s) :",
            Dep[1:],
            ["Paris"])
        if len(options_bar_plot) != 0:
            data_filtre_bar_plot = data[data['Département'].isin(options_bar_plot)]
            figure_barplot = plt.figure(figsize=(8, 6))
            sns.barplot(x=selected_type_bar_plot, y="prix_valeur", hue="Essence", data=data_filtre_bar_plot)
            plt.xlabel(str(selected_type_bar_plot))
            plt.ylabel("Eur/L")
            plt.title("Prix de l'essence moyen à la pompe")
            plt.xticks(rotation=45)
            st.pyplot(figure_barplot)
    else:
        options_bar_plot = st.multiselect(
            'Sélectionne une ou des ' + str(selected_type_bar_plot) + "(s) :",
            Ville[1:],
            ["Paris"])
        if len(options_bar_plot) != 0:
            data_filtre_bar_plot = data[data['Ville'].isin(options_bar_plot)]
            figure_barplot = plt.figure(figsize=(8, 6))
            sns.barplot(x=selected_type_bar_plot, y="prix_valeur", hue="Essence", data=data_filtre_bar_plot)
            plt.xlabel(str(selected_type_bar_plot))
            plt.ylabel("Eur/L")
            plt.title("Prix de l'essence moyen à la pompe")
            plt.xticks(rotation=45)
            st.pyplot(figure_barplot)



if __name__== "__main__":
    data , date_la_plus_recente, Ville ,Dep ,Region, Essence_liste = load_clean_data(link)
    main(data, date_la_plus_recente, Ville, Dep, Region,Essence_liste)

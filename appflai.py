import streamlit as st
import pandas as pd 
from pycaret.classification import load_model, predict_model

st.set_page_config(page_title = 'App FLAI - Powered by Streamlit', 
				   page_icon = 'iconeflai.png' ,
				   layout = 'centered', 
				   initial_sidebar_state = 'auto')

modelo = load_model('modelo-para-previsao-de-salario')

@st.cache
def ler_dados():
	dados = pd.read_csv('prof-dados-resumido.csv')
	dados = dados.dropna()
	return dados

dados = ler_dados()  

st.image('bannerflai.jpg', use_column_width = 'always')

st.sidebar.write('''
# :sparkles: **App [FLAI](https://www.flai.com.br/)**
***Powered by [Streamlit](https://streamlit.io/)***

---

	''')
 
opcoes = paginas = ['Home', 'Modelo de Proposta de Salário', 'Sobre']
pagina = st.sidebar.radio('Selecione uma página:', paginas)
#st.sidebar.markdown('---')

if pagina == 'Home':
	

	st.write("""
	# :sparkles: Bem-vindo ao App FLAI - O Salário do Profissional de Dados
	***Powered by Streamlit***
	
	---

	Nesse Web App vamos fazer análises de dados rápidas, dashboards e deploy de um modelo para estimar salários de profissionais da área de dados.

	### Funcionalidades no momento
	
	:ballot_box_with_check:  Página Inicial: Home
	
	:ballot_box_with_check:  Modelo de Estimação de Salário de Profissionais de Dados no Brasil
		
	:ballot_box_with_check:  Página Sobre 

	Os modelos desse web-app foram desenvolvidos utilizando o conjunto de 
	dados que pode ser encontrado nesse [link do kaggle](https://www.kaggle.com/datahackers/pesquisa-data-hackers-2019).
			
	Caso encontre algum erro/bug, por favor, não hesite em entrar em contato! :poop:
	
	Para mais informações sobre o Streamlit, consulte o [site oficial](https://www.streamlit.io/) ou a sua [documentação](https://docs.streamlit.io/_/downloads/en/latest/pdf/).
	 

	""")



if pagina == 'Modelo de Proposta de Salário': 
	st.markdown('---')
	st.markdown('## **Modelo para Estimar o Salário de Profissionais da área de Dados**')
	st.markdown('Utilize as variáveis abaixo para utilizar o modelo de previsão de salários desenvolvido [aqui]().')
	st.markdown('---')

	col1, col2, col3 = st.beta_columns(3)

	x1 = col1.radio('Idade', dados['Idade'].unique().tolist() )
	x2 = col1.radio('Profissão', dados['Profissão'].unique().tolist())
	x3 = col1.radio('Tamanho da Empresa', dados['Tamanho da Empresa'].unique().tolist())
	x4 = col1.radio('Cargo de Gestão', dados['Cargo de Gestão'].unique().tolist())
	x5 = col3.selectbox('Experiência em DS', dados['Experiência em DS'].unique().tolist()) 
	x6 = col2.radio('Tipo de Trabalho', dados['Tipo de Trabalho'].unique().tolist() )
	x7 = col2.radio('Escolaridade', dados['Escolaridade'].unique().tolist())
	x8 = col3.selectbox('Área de Formação', dados['Área de Formação'].unique().tolist())
	x9 = col3.selectbox('Setor de Mercado', dados['Setor de Mercado'].unique().tolist())
	x10 = 1
	x11 = col2.radio('Estado', dados['Estado'].unique().tolist()) 
	x12 = col3.radio('Linguagem Python', dados['Linguagem Python'].unique().tolist()) 
	x13 = col3.radio('Linguagem R', dados['Linguagem R'].unique().tolist()) 
	x14 = col3.radio('Linguagem SQL', dados['Linguagem SQL'].unique().tolist()) 
	 

	dicionario  =  {'Idade': [x1],
				'Profissão': [x2],
				'Tamanho da Empresa': [x3],
				'Cargo de Gestão': [x4],
				'Experiência em DS': [x5],
				'Tipo de Trabalho': [x6],
				'Escolaridade': [x7],
				'Área de Formação': [x8],
				'Setor de Mercado': [x9],
				'Brasil': [x10],
				'Estado': [x11],		
				'Linguagem Python': [x12],
				'Linguagem R': [x13],
				'Linguagem SQL': [x14]}

	dados = pd.DataFrame(dicionario)  

	st.markdown('---') 
	st.markdown('## **Quando terminar de preencher as informações da pessoa, clique no botão abaixo para estimar o salário de tal profissional**') 


	if st.button('EXECUTAR O MODELO'):
		saida = float(predict_model(modelo, dados)['Label']) 
		st.markdown('## Salário estimado de **R$ {:.2f}**'.format(saida))






if pagina == 'Sobre':

	st.markdown(""" 

	## **Sobre**

	Nesse Web App mostramos o poder do streamlit para construir soluções fáceis, rápidas e que permitem uma usabilidade bastante ampla.')
		
	Pare por um momento e imagine o universo de possibilidades que temos ao combinar\
		todos os recursos do streamlit com o que já temos no Python.
		
	Esse tipo de web-app é perfeito para quando se quer entregar uma solução rápida\
		e/ou criar um ambiente de testes mais eficiente.

	Não deixe de explorar os recursos do streamlit. Aprenda, crie, desenvolva. \
		Faça o que ninguém fez ainda. Vá além. 

	*#itstimetoflai* :rocket:

	---
	
             """) 

	if st.button('Comemorar'):
		st.balloons()





st.sidebar.markdown('---')
st.sidebar.image('logoflai.png', width = 90)




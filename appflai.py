import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pycaret.classification import load_model, predict_model

st.set_page_config(page_title = 'App FLAI - Powered by Streamlit',  
				   layout = 'centered', 
				   initial_sidebar_state = 'auto')

modelo = load_model('modelo-para-previsao-de-salario')

@st.cache
def ler_dados():
	dados = pd.read_csv('prof-dados-resumido.csv')
	dados = dados.dropna()
	return dados

dados = ler_dados()  

st.sidebar.markdown('# **App FLAI - Powered by *Streamlit***')
opcoes = paginas = ['Home', 'Análise de Dados', 'Dashboard', 'Modelo de Proposta de Salário', 'Streamlit Widgets', 'Sobre']
pagina = st.sidebar.radio('Selecione uma página:', paginas)

if pagina == 'Home':
	st.image('bannerflai.jpg', use_column_width = 'always')

	st.write("""
	# Bem-vindo ao App FLAI - Powered by Streamlit
	
	Nesse Web App podemos utilizar em produção os modelos desenvolvidos tanto para
	precificar novos seguros, quanto para buscar por fraudadores do seguro.
	
	A lista abaixo ilustra o que está implementado até o momento. 
	### Funcionalidades no momento
	
	:ballot_box_with_check:  Página Inicial 
	
	:ballot_box_with_check:  Modelo em produção para precificar planos de saúde em novos clientes
	
	:ballot_box_with_check:  Modelo em produção para detectar possíveis fraudadores 
	
	:black_square_button:  Deploy em lote (vários pessoas ao mesmo tempo)
	
	:ballot_box_with_check:  Página de créditos 

	Os modelos desse web-app foram desenvolvidos utilizando o conjunto de 
	dados que pode ser encontrado nesse [link do kaggle](https://www.kaggle.com/mirichoi0218/insurance).
	
	O referencial sobre os modelos utilizados você pode encontrar nesse [link](https://github.com/gitflai/Workshop-DDS/blob/main/Dados_de_Custos_Medicos.ipynb). 
	
	Os modelos são desenvolvidos e analisados utilizando a biblioteca [PyCaret](https://pycaret.org/).
	
	Caso encontre algum erro/bug, por favor, não hesite em entrar em contato! :poop:
	
	Para mais informações sobre o Streamlit, consulte o [site oficial](https://www.streamlit.io/) ou a sua [documentação](https://docs.streamlit.io/_/downloads/en/latest/pdf/).
	
	[Lista de emojis para markdown](https://gist.github.com/rxaviers/7360908).
 
	""")





if pagina == 'Análise de Dados': 
	variaveis = dados.columns.to_list()

	var = st.selectbox('Selecione:', variaveis) 
	g1  = dados[var].value_counts().plot(kind = 'barh', title = 'Contagem {}'.format(var)) 
	st.pyplot(g1.figure)
 
	st.markdown('---')
	lvar2 = variaveis.copy()
	lvar2.pop(0) 
	var1 = st.selectbox('Selecione:', lvar2)
	g2 = dados['Salário'].groupby(dados[var1]).mean().plot(kind = 'barh', title = 'Salário por {}'.format(var1))
	st.pyplot(g2.figure)


	st.markdown('---')
	v1 = st.selectbox('Selecione uma variável:', lvar2)
	v2 = st.selectbox('Selecione outra variável:', lvar2)
	titulo = 'Salário por {} e {}'.format(v1, v2)
	g3 = dados.groupby([v1, v2]).mean()['Salário'].unstack().plot(kind = 'barh', title = titulo)
	st.pyplot(g3.figure)


if pagina == 'Dashboard': 

	prof = st.sidebar.radio('Profissão', dados['Profissão'].unique().tolist(), index = 3)
	dados0 = dados[dados['Profissão'] == prof]
	n = dados0.shape[0]
	s = dados0['Salário'].mean()

	st.markdown('# Dashboard dos **{}**'.format(prof))

	st.markdown('---')
	col1, col2 = st.beta_columns((1, 2))
	col1.markdown('### Amostra: **{}**'.format(n))
	col2.markdown('### Salário: **R${:.2f}**'.format(s)) 

	st.markdown('---')
	col1, col2 = st.beta_columns((1, 2))

	d1 = dados0['Escolaridade'].value_counts().plot(kind = 'pie')
	col1.pyplot(d1.figure, clear_figure = True)

	d2 = dados0['Linguagem Python'].value_counts().plot(kind = 'bar', title ='Python') 
	col1.pyplot(d2.figure, clear_figure = True)

	titulo = 'Salário por Idade e Tamanho da Empresa'
	d3 = dados0.groupby(['Idade', 'Tamanho da Empresa']).mean()['Salário'].unstack().plot(kind = 'barh', title = titulo)
	col2.pyplot(d3.figure, clear_figure = True)

	st.markdown('---')


if pagina == 'Modelo de Proposta de Salário': 
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
	
	st.markdown('---')

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

	if st.button('EXECUTAR O MODELO'):
		saida = float(predict_model(modelo, dados)['Label']) 
		st.markdown('## Salário estimado de **R$ {:.2f}**'.format(saida))









if pagina == 'Streamlit Widgets':
	# col1, col2 = st.beta_columns(2) 
	st.markdown('---')

	st.markdown('### **Botões**')
	st.markdown('Guardam valores **True** ou **False**')
	st.code("st.button(label = '-> Clique aqui! <-', help = 'É só clicar ali')")
	st.button(label = '-> Clique aqui! <-', help = 'É só clicar ali')

	st.markdown('---')

	st.markdown('### **Caixa de Selecionar**')
	st.markdown('Guardam valores **True** ou **False**')
	st.code("st.checkbox('Clique para me selecionar', help = 'Clique e desclique quando quiser')")
	st.checkbox('Clique para me selecionar', help = 'Clique e desclique quando quiser')

	st.markdown('---')

	st.markdown('### **Botões de Rádio**')
	st.markdown('Guarda o item do botão selecionado')
	st.code("st.radio('Botões de Rádio', options = [100, 'Python', print, [1, 2, 3]], index = 1, help = 'Ajuda')")
	st.radio('Botões de Rádio', options = [100, 'Python', print, [1, 2, 3]], index = 1, help = 'Ajuda')

	st.markdown('---')

	st.markdown('### **Caixas de Seleção**')
	st.markdown('Guarda o item da caixa selecionado')
	st.code("st.selectbox('Clique no item que deseja', options = ['azul', 'roxo', 'verde'], index = 2)")
	st.selectbox('Clique no item que deseja', options = ['azul', 'roxo', 'verde'], index = 2)

	st.markdown('---')

	st.markdown('### **Caixas de Seleção Múltipla**')
	st.markdown('Guarda a lista de itens selecionados')
	st.code("st.multiselect('Selecione quantas opções desejar', options = ['A', 'B', 'C', 'D', 'E'])")
	st.multiselect('Selecione quantas opções desejar', options = ['A', 'B', 'C', 'D', 'E'])
	
	st.markdown('---')

	st.markdown('### **Slider**')
	st.markdown('Guarda o número selecionado')
	st.code("st.slider('Entrada numérica', min_value = 1, max_value = 25, value = 7, step = 2)	")
	st.slider('Entrada numérica', min_value = 1, max_value = 25, value = 7, step = 2)	
	
	st.markdown('---')

	  
	st.select_slider('Slide to select', options=[1,'2'])
	st.text_input('Enter some text')
	st.number_input('Enter a number')
	st.text_area('Area for textual entry')
	st.date_input('Date input')
	st.time_input('Time entry')
	st.file_uploader('File uploader')
	st.color_picker('Pick a color')


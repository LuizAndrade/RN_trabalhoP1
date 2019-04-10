# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 15:19:59 2018

@author: Karla Figueiredo, Hugo Lima, Thaina Figueiredo
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 16:08:38 2018

"""
import time as t
from keras import losses
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.optimizers import Nadam
from keras import optimizers
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from math import sqrt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import shift

# be able to save images on server
matplotlib.use('Agg')
from matplotlib import pyplot

# date-time parsing function for loading the dataset
# lendo a base de dados
def parser(x):
    return datetime.strptime('20'+x, '%Y-%m')

#Janelamento dataset de Treino
def cria_matriz_treino(df_scaled):
    lag = 13
    df_aux = df_scaled[lag-13:lag]
    tam_conj_treino = len(df_scaled) - lag
    i = 0
    df_treino = DataFrame(cria_variavel_endogena_treino(df_aux,i))
    for i in range(tam_conj_treino):
        df_aux[i+1] = df_aux[i].shift(-1)
        df_aux.iat[12,i+1] = df_scaled.iat[lag,0]
        df_treino[i+1] = cria_variavel_endogena_treino(df_aux,i+1)
        lag = lag + 1
    return df_treino

def cria_matriz_validacao(df_scaled):
    lag = 13
    i=0

    df_aux = df_scaled[lag-13:lag]
    tam_conj_validacao = len(df_scaled)-lag
    df_validacao = DataFrame(df_aux)

    for i in range(tam_conj_validacao):
        #chamada a função que faz a previsão para pegar o retorno dela e incluir no vetor
        df_aux[i+1]= df_aux[i].shift(-1)
        df_aux.iat[12,i+1] = df_scaled.iat[lag,0]
        df_validacao[i+1] = df_aux[i+1]# cria_variavel_endogena(df_aux,i+1) #chamada a função que cria as variáveis
        lag = lag+1
    return df_validacao


# Faz o multi step, ou seja, inclui o valor previso na entrada para a proxima previsão. E adiciona o target 
# na ultima posição. Além de fazer o janelamento.
def cria_vetor_validacao(valida_recorrente,valor_previsto,validation_scaled,i):
    b = validation_scaled
    aux = valida_recorrente.T
    df1 = DataFrame(aux)
    df1[i+1] = df1[i].shift(-1)
    df1.iat[11,i+1] = valor_previsto
    df1.iat[12,i+1] = b[i+1]
    aux_M = df1.as_matrix()    
    vetor_validacao = aux_M.T  
    return vetor_validacao

def cria_matriz_teste(df_scaled):   
    #pos_inicial = 174
    #pos_final = 198
    lag = 13
    i=0
    df_aux = df_scaled[lag-13:lag]
    tam_conj_teste = len(df_scaled)-lag
    df_teste = DataFrame(df_aux)

    for i in range(tam_conj_teste): 
        #chamada a função que faz a previsão para pegar o retorno dela e incluir no vetor
        df_aux[i+1]= df_aux[i].shift(-1)
        df_aux.iat[12,i+1] = df_scaled.iat[lag,0]
        df_teste[i+1] = df_aux[i+1] #DataFrame(cria_variavel_endogena(df_aux,i+1)) #chamada a função que cria as variáveis
        lag = lag + 1
    return df_teste

def cria_variavel_endogena_treino(df_variavel, i): 
    D12 = df_variavel.iat[11,i] - df_variavel.iat[10,i] # Diferença entre o ultimo e penultimo valor
    D13 = df_variavel.iat[11,i] - df_variavel.iat[9,i]
    MM2 = (df_variavel.iat[11,i] + df_variavel.iat[10,i])/2
    MM3 = (df_variavel.iat[11,i] + df_variavel.iat[10,i] + df_variavel.iat[9,i])/3
    MM6 = (df_variavel.iat[11,i] + df_variavel.iat[10,i] + df_variavel.iat[9,i] + df_variavel.iat[8,i] + df_variavel.iat[7,i] + df_variavel.iat[6,i])/6
    MM12 = (df_variavel.iat[11,i] + df_variavel.iat[10,i] + df_variavel.iat[9,i] + df_variavel.iat[8,i] + df_variavel.iat[7,i] + df_variavel.iat[6,i] + df_variavel.iat[5,i] + df_variavel.iat[4,i] + df_variavel.iat[3,i] + df_variavel.iat[2,i] + df_variavel.iat[1,i] + df_variavel.iat[0,i])/12
    M1 = df_variavel.iat[11,i]
    M2 = df_variavel.iat[10,i]
    M3 = df_variavel.iat[9,i]
    M4 = df_variavel.iat[8,i]
    M6 = df_variavel.iat[6,i]
    M12 = df_variavel.iat[0,i]
    SR = df_variavel.iat[12,i]
    aux = [D13, D12, MM12, MM6, MM3, MM2, M12, M6, M4, M3, M2, M1, SR]
    vetor_var = list(aux) 
    return vetor_var

def cria_variavel_endogena(df_variavel, i=0):
    D12 = df_variavel[i,11] - df_variavel[i,10] # Diferença entre o ultimo e penultimo valor , "Diferença " ,D12
    D13 = df_variavel[i,11] - df_variavel[i,9]
    MM2 = (df_variavel[i,11] + df_variavel[i,10])/2
    MM3 = (df_variavel[i,11] + df_variavel[i,10] + df_variavel[i,9])/3
    MM6 = (df_variavel[i,11] + df_variavel[i,10] + df_variavel[i,9] + df_variavel[i,8] + df_variavel[i,7] + df_variavel[i,6])/6
    MM12 = (df_variavel[i,11] + df_variavel[i,10] + df_variavel[i,9] + df_variavel[i,8] + df_variavel[i,7] + df_variavel[i,6] + df_variavel[i,5] + df_variavel[i,4] + df_variavel[i,3] + df_variavel[i,2] + df_variavel[i,1] + df_variavel[i,0])/12
    M1 = df_variavel[i,11]
    M2 = df_variavel[i,10]
    M3 = df_variavel[i,9]
    M4 = df_variavel[i,8]
    M6 = df_variavel[i,6]
    M12 = df_variavel[i,0]
    SR = df_variavel[i,12]
    vetor_var = [D13, D12, MM12, MM6, MM3, MM2, M12, M6, M4, M3, M2, M1, SR]
    vetor_var_M = np.reshape(vetor_var,(1,-1))
    return vetor_var_M


# create a differenced series
# fazendo o diferencial da série para remover a tendência
# 
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# scale train and test data to [-1, 1]
# normalizando os dados [-1, 1]
    
def scale(df_transformed):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # Pega o min e max de todo o conjunto de dados e normaliza de acordo com ele.
    df_scaler = scaler.fit_transform(df_transformed)
    return scaler, df_scaler #scaler, train_scaled, validation_scaled, test_scaled

# inverse scaling for a forecasted value
# "desnormalizando"os dados
def invert_scale(scaler, X, yhat):
    new_row = [x for x in X] + [yhat]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# evaluate the model on a dataset, returns RMSE in transformed units
def evaluate_rmse(model, raw_data, scaled_dataset, scaler, offset, n_batch):
    # separate
    X, y = scaled_dataset[:,0:-1], scaled_dataset[:,-1]
    # forecast dataset
    output = model.predict(X, batch_size=n_batch)
    # invert data transforms on forecast
    predictions = list()
    for i in range(len(output)):
        yhat = output[i,0]
        # invert scaling
        yhat = invert_scale(scaler, X[i], yhat)
        # invert differencing
        yhat = yhat + raw_data[i]
        # store forecast
        predictions.append(yhat)
    # report performance
    
    rmse = sqrt(mean_squared_error(raw_data[:], predictions))
    return rmse
        
# evaluate the model on a validation dataset, returns RMSE in transformed units
def evaluate_val_rmse(model, raw_data, scaled_dataset, scaler, offset, n_batch):
    # separate
    X, y = np.array(scaled_dataset)[:,0:-1], np.array(scaled_dataset)[:,-1]
    # forecast dataset
    output = model.predict(X, batch_size=n_batch)
    # invert data transforms on forecast
    predictions = list()
    for i in range(len(output)):
        yhat = output[i,0]
        # invert scaling
        yhat = invert_scale(scaler, X[i], yhat)
        # invert differencing
        yhat = yhat + raw_data[i]
        # store forecast
        predictions.append(yhat)
    # report performance
    
    rmse = sqrt(mean_squared_error(raw_data[:], predictions))
    return rmse

def evaluate_rmse_semescala(model, raw_data, scaled_dataset, scaler, offset, n_batch):
    # separate
    X, y = scaled_dataset[:,0:-1], scaled_dataset[:,-1]
    # forecast dataset
    output = model.predict(X, batch_size=n_batch)
    # invert data transforms on forecast
    predictions = list()
    for i in range(len(output)):
        yhat = output[i,0]
        # store forecast
        predictions.append(yhat)
    # report performance
    
    #rmse = sqrt(mean_squared_error(y, output))
    mse = (mean_squared_error(y, output))

    return mse
    
# evaluate the model on a dataset, returns RMSE in transformed units
def evaluate_mape(model, raw_data, scaled_dataset, scaler, offset, n_batch):
    # separate
    X, y = scaled_dataset[:,0:-1], scaled_dataset[:,-1]
    # forecast dataset
    output = model.predict(X, batch_size=n_batch)
    # invert data transforms on forecast
    predictions = list()
    for i in range(len(output)):
        yhat = output[i,0]
        # invert scaling
        yhat = invert_scale(scaler, X[i], yhat)
        # invert differencing
        yhat = yhat + raw_data[i]
        # store forecast
        predictions.append(yhat)
    # report performance
    mape = np.mean(np.abs((raw_data[:] - predictions) / raw_data[:])) * 100
    return mape

def evaluate_mse_semescala(model, raw_data, scaled_dataset, scaler, offset, n_batch):
    # separate
    X, y = scaled_dataset[:,0:-1], scaled_dataset[:,-1]
    # forecast dataset
    output = model.predict(X, batch_size=n_batch)
    # invert data transforms on forecast
    predictions = list()
    for i in range(len(output)):
        yhat = output[i,0]
        predictions.append(yhat)
    # report performance
    mape = np.mean(np.abs((y - output) / y)) * 100
    return mape

def evaluate_test(model, raw_data, scaled_dataset, scaler, offset, n_batch):
    # separate

    X, y = scaled_dataset[:,0:-1], scaled_dataset[:,-1]
    # forecast dataset
    output = model.predict(X, batch_size=n_batch)
    # invert data transforms on forecast
    predictions = list()
    for i in range(len(output)):
        yhat = output[i,0]
        # invert scaling
        yhat = invert_scale(scaler, X[i], yhat)
        # invert differencing
        yhat = yhat + raw_data[i]
        # store forecast
        predictions.append(yhat)
    # report performance
    return predictions


def evaluate_single_step(model, scaled_dataset):
    X, y = scaled_dataset[:,0:-1], scaled_dataset[:,-1]
    # forecast dataset
    output = model.predict(X, batch_size=1)
  
    return output


# fit an MLP network to training data
inicio = t.time()
def modelo(train, validation, test, raw, scaler, n_batch, nb_epoch, neurons):
    # Separando o dataset de treino em entradas e target
    X, y = train[:, 0:-1], train[:, -1]
    model = Sequential()
    # inicializacao dos neuronios
   # kernel_initializer='random_normal') 
   # keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
   # kernel_initializer='glorot_uniform', distribuicao normal centrada no zero e truncada por 
   # stddev = sqrt(2 / (entradas do neuronio + saidas do neuronio))
   
# definicao da funcao de ativacao
   # activation= 'tanh'
   # activation= 'sigmoid'
   # activation= 'linear'
   # activation= 'softmax'
   # activation= 'softsign' -> x / (abs(x) + 1
    #neurons = 2  
    model.add(Dense(neurons, activation='tanh', input_dim=X.shape[1]))
    model.add(Dense(1))

#definicao da estrategia de aprendizado
# optimizer= 'sgd'
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    # keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #opt_Nadam=Nadam(lr=0.001, beta_1=0.8, beta_2=0.9, epsilon=None, schedule_decay=0.004)

   
    # definicao da funcao erro
    # keras.losses.mean_absolute_error(y_true, y_pred) (MAE)
    # keras.losses.mean_absolute_percentage_error(y_true, y_pred) (MAPE)

    train_rmse, valida_rmses, test_rmse, melhor_arq_rmse = list(), list(), list(), list()
    train_mape, valida_mapes, test_mape = list(), list(), list()
    test_predict=[]
    validation_predict=[]
    #valida_mse=[]
    valores_previstos=[]
    valores_previstos_test = []
    
    model.compile(loss='mean_squared_error', optimizer=sgd, 
                  metrics=['mse', 'mae', 'mape'])    
    
    #earlystop = EarlyStopping(monitor='val_loss',
    #                          min_delta=0.0001,
    #                          patience=0,
    #                          verbose=0, mode='auto')
    
   # patience = representa o número de épocas antes de parar assim que sua perda 
   #            começar a aumentar (para de melhorar). 
   #            Com treino em batch e uma pequena taxa de aprendizado, sua perda será mais suave, 
   #            então  deve se usar  patience  menor. 
   
    raw_train = raw[-(len(train)+len(validation)+len(test)):-(len(test)+len(validation))]    
    raw_validation = raw[-(len(validation)+len(test)):-len(test)]
    raw_test = raw[-len(test):]

    pior_erro_rmse = 0
    count_pior_erro = 0
    melhor_erro_rmse = 99999999999999999
    num_epocas= 0 # será um numero infinitamente grande para que o treinamento seja interrompido exclusivamente pela curva
    # do erro de validação.
    # época - numero de  vezes que a rede será treinada
    for num_epocas in range(nb_epoch):
        # Treinando o modelo apenas com o dataset de treino
        history=model.fit(X, y, epochs=num_epocas, batch_size=n_batch, verbose=0, shuffle=False)

        # Pega os primeiros 12 valores do conjunto de validação
        valida_recorrente = validation[:1,:]
        #valida_recorrente = validation_scaled[:1,:]
        # Manipulando os valores de entrada do conjunto de validação
        valida_recorrente_modificado = cria_variavel_endogena(valida_recorrente,0)
        validacao_modificado=valida_recorrente_modificado
        i=0
        for i in range(len(validation)-1):
            # a funçao abaixo realiza a previsao de um valor para uma entrada do conj. de validação
            valor_previsto=evaluate_single_step(model,valida_recorrente_modificado)
            # Lista com os valores previstos
            valores_previstos.append(valor_previsto)
            # Incluindo o valor previsto na entrada da proxima época
            valida_recorrente=cria_vetor_validacao(valida_recorrente,valor_previsto,validation[:,-1],i)
            #valida_recorrente=cria_vetor_validacao(valida_recorrente,valor_previsto,validation_scaled[:,-1],i)
            # manipulando as entradas do conjunto de validação, agora com o valor previsto.
            valida_recorrente_modificado= cria_variavel_endogena(valida_recorrente[-1:,:],0)
            # Guardando os vetores de entrada do conjunto de validação
            validacao_modificado=np.append(validacao_modificado,valida_recorrente_modificado, axis=0)

        # evaluate model on train data
        train_rmse.append(evaluate_rmse(model, raw_train, train, scaler, 0, n_batch))
        train_mape.append(evaluate_mape(model, raw_train, train, scaler, 0, n_batch))   
        
        # Calcula o erro RMSE para a configuração de modelo treinada quando inputado a entrada de validação
        valida_rmses.append(evaluate_rmse(model, raw_validation, validacao_modificado, scaler, 0, n_batch))
        # Calcula o erro MAPE para a configuração de modelo treinada quando inputado a entrada de validação
        valida_mapes.append(evaluate_mape(model, raw_validation, validacao_modificado, scaler, 0, n_batch))
        # '''
        # valores_previstos
        # predictions = list()
        # for i in range(len(valores_previstos)):
        # yhat = valores_previstos[i,0]
        # # invert scaling
        # yhat = invert_scale(scaler, X[i], yhat)
        # # invert differencing
        # yhat = yhat + raw_data[i]
        # # store forecast
        # predictions.append(yhat)
        # '''
       
        test_recorrente = test[:1,:]

        # Manipulando os valores de entrada do conjunto de teste
        test_recorrente_modificado = cria_variavel_endogena(test_recorrente,0)
        test_modificado=test_recorrente_modificado
        j=0
        for j in range(len(test)-1):
            # a funçao abaixo realiza a previsao de um valor para uma entrada do conj. de teste
            valor_previsto_test=evaluate_single_step(model,test_recorrente_modificado)
            # Lista com os valores previstos de teste 
            valores_previstos_test.append(valor_previsto_test)
            # Incluindo o valor previsto na entrada da proxima época
            test_recorrente=cria_vetor_validacao(test_recorrente,valor_previsto_test,test[:,-1],j)
            #valida_recorrente=cria_vetor_validacao(valida_recorrente,valor_previsto,validation_scaled[:,-1],i)
            # manipulando as entradas do conjunto de teste, agora com o valor previsto.
            test_recorrente_modificado= cria_variavel_endogena(test_recorrente[-1:,:],0)
            # Guardando os vetores de entrada do conjunto de teste
            test_modificado=np.append(test_modificado,test_recorrente_modificado, axis=0)
            
        test_rmse.append(evaluate_rmse(model, raw_test, test_modificado, scaler, 0, n_batch))
        test_mape.append(evaluate_mape(model, raw_test, test_modificado, scaler, 0, n_batch))
        # Verificando se o erro de validação encontrado por esta época é menor que o da época anterior.
        # Se for, esse passa a ser o menor erro. Senao, o erro anterior é o menor e então paramos o treinamento.
        if valida_rmses[num_epocas] <= melhor_erro_rmse: # If 1 - Verifica o melhor erro de validação
            #print("Entrei no if 1")
            melhor_erro_rmse = valida_rmses[num_epocas] 
            melhor_arq_rmse.append(evaluate_rmse(model, raw_test, test_modificado, scaler, 0, n_batch))
            test_predict=evaluate_test(model, raw_test, test_modificado, scaler, 0, n_batch)
            validation_predict=evaluate_test(model, raw_validation, validacao_modificado, scaler, 0, n_batch)
            count_pior_erro = 0
        else: 
            if count_pior_erro == 10: # If 2 - Controla o numero de erros de validação sequenciais com aumento
                break
            else:
                if valida_rmses[num_epocas] >= pior_erro_rmse: # if 3 - verifica se o erro de validação encontrado
                    #continua a aumentar, ou seja, garante que o aumento é sequencial.
                    count_pior_erro = count_pior_erro+1
                    pior_erro_rmse = valida_rmses[num_epocas]
                else: # if 4 - é o caso onde ocorre um erro menor em uma sequencia de erros com aumento. Neste caso,
                    # zero o contador e começo a contar de novo.
                    #valida_rmses[num_epocas] < pior_erro_rmse
                    count_pior_erro = 0
                    pior_erro_rmse = valida_rmses[num_epocas]
        
        
        
      
    # '''
    # valor_previsto=evaluate_single_step(model,valida_recorrente_modificado)
    #         valores_previstos.append(valor_previsto)
    #         valida_recorrente=cria_vetor_validacao(valida_recorrente,valor_previsto,validation_scaled[:,-1],i)
    #         validacao_modificado.append(valida_recorrente_modificado)
    #         valida_recorrente_modificado= cria_variavel_endogena(valida_recorrente[-1:,:],0) 
    # Ao final desse for terá todos os valores previstos durante a validacao.
    # esse caso deveria calcular o MSE para esses valores e decidir se interrompe o treinamento ou 
    # seja, se interrompe o for que está alinhado nesse contexto.
    # Para isso precisaria controlar os erros MSE que está obendendo sequencialmente
    # com os valores de erro, poderia criar um vetor de erros de validacao que obteria a cada epoca.

    # '''
    # ---------------------------
    #print("Tamanho conjunto teste:" , len(test))
    print("Valor Real Validação: ", raw_validation)
    print("Valores Previstos Validação: ", validation_predict)
    print("Somatório dos Valores Previstos Val 1: ", sum(validation_predict[0:12]), "\nSomatório dos Valores Previstos Val 2: ", sum(validation_predict[12:24]))
    
    print("Valor Real Teste:\n ", raw_test)
    print("Valores Previstos Teste:\n ", test_predict)
    print("Somatório dos Valores Previstos Teste: ", sum(test_predict))
    
    # Printando o erro RMSE de validação e treino
    plt.plot(train_rmse, color='red') #Criando o gráfico
    plt.plot(valida_rmses, color='blue') #Criando o gráfico
    plt.plot(test_rmse, color='green') #Criando o gráfico
    plt.title('TREINO VS VALIDAÇÃO VS TESTE') #adicionando o título
    plt.xlabel('Épocas')
    plt.ylabel('RMSE')
    plt.legend(['train', 'val', 'test'], loc='best')
    plt.show()
    
    # Printando o erro MAPE de validação E TREINO
    plt.plot(train_mape, color='red') #Criando o gráfico
    plt.plot(valida_mapes, color='blue') #Criando o gráfico
    plt.plot(test_mape, color='green') #Criando o gráfico
    plt.title('TREINO VS VALIDAÇÃO VS TESTE') #adicionando o título
    plt.xlabel('Épocas')
    plt.ylabel('MAPE')
    plt.legend(['train', 'val', 'test'], loc='best')
    plt.show()
    '''
    # Printando os valores previstos na validação
    plt.plot(valores_previstos, color='red') #Criando o gráfico
    plt.plot(raw_validation, color='blue') #Criando o gráfico
    plt.title('PREVISTOS - VALIDAÇÃO') #adicionando o título
    plt.xlabel('Meses')
    plt.ylabel('ICMS')
    plt.show() 
    '''
    # Printando os valores previstos no teste
    plt.plot(test_predict, color='red') #Criando o gráfico
    plt.plot(raw_test, color='blue') #Criando o gráfico
    plt.title('Previsões - Teste') #adicionando o título
    plt.xlabel('Meses')
    plt.ylabel('ICMS')
    plt.legend(['Previsto', 'Real'], loc='best')
    plt.show()    
    
    hist_rmse=DataFrame()
    hist_mape=DataFrame()
    melhor_arq=DataFrame()
    
    # revisão Karla
    #---------------------------
    hist_rmse['train'],hist_rmse['validation']= train_rmse, valida_rmses
    hist_mape['train'],hist_mape['validation']= train_mape, valida_mapes
    melhor_arq['melhor arq']= melhor_arq_rmse
    return hist_rmse, hist_mape,melhor_arq


fim = t.time()
print("Tempo de Execução: ", (fim - inicio))

# run diagnostic experiments
def run(n_epochs,nh, repeats, n_batch):
    # Carregando os dados da planilha de origem
    series = read_csv('ICMS-RJ.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    raw_values = series.values
    # Fazendo o diferencial na serie
    diff_values = difference(raw_values, 1) 
    # DataFrame com todos os valores já com o diferencial
    df = DataFrame(diff_values)
    #print("\nDataFrame Diferenciado\n", df)
    scaler,scaled = scale(df)
    df_scaled = DataFrame(scaled)

    #Criando as váriaveis endógenas de cada conjunto
    train, validation, test = cria_matriz_treino(df_scaled[:-36]), cria_matriz_validacao(df_scaled[-48:-12]), cria_matriz_teste(df_scaled[-24:])

    # Transformando os conj. de treino, validação e teste em um unico array para normalizar
    train_M = train.as_matrix()
    train_M = train_M.T
    train_M = np.reshape(train_M,(-1,1))
    validation_M = validation.as_matrix()
    validation_M = validation_M.T
    validation_M = np.reshape(validation_M,(-1,1))
    test_M = test.as_matrix()
    test_M = test_M.T
    test_M = np.reshape(test_M,(-1,1))
     

    # Criando os arrays de entrada de 13 posições para cada conjunto, onde cada linha é a entrada da rede
    train_scaled, validation_scaled,test_scaled = np.reshape(train_M,(-1,13),order = 'C'), np.reshape(validation_M,(-1,13),order = 'C'), np.reshape(test_M,(-1,13),order = 'C')
     
    # run diagnostic tests
    error_train_scores_rmse = []
    error_train_scores_mape=[]
    error_validation_scores_rmse=[]
    error_validation_scores_mape=[]
    error_test_scores_rmse=[]
    error_test_scores_mape=[]
    
    history_rmse, history_mape, melhor_arq = DataFrame(),DataFrame(),DataFrame()

    # o repeat equivale ao numero de redes que seráo instanciadas. Dentro do numero de instancias da rede, tem o numero
    # de epocas, que equivale quantas vezes a rede será treinada.
    for i in range(repeats):
        history_rmse,history_mape,melhor_arq = modelo(train_scaled, validation_scaled, test_scaled, raw_values, scaler, n_batch, n_epochs, nh)        

       
        #revisao Karla
        #---------------------------
        print('%d) TrainRMSE=%f, ValidationRMSE=%f, TestRMSE=%f' % (i, history_rmse['train'].iloc[-1], history_rmse['validation'].iloc[-1], melhor_arq['melhor arq'].iloc[-1]))

        
        # revisao Karla
        # ---------------------------
        print('%d) TrainMAPE=%f, ValidationMAPE=%f' % (i, history_mape['train'].iloc[-1], history_mape['validation'].iloc[-1]))
        
        error_train_scores_rmse.append(history_rmse['train'].iloc[-1])
        error_train_scores_mape.append(history_mape['train'].iloc[-1])
 
        # revisão Karla
        error_validation_scores_rmse.append(history_rmse['validation'].iloc[-1])
        error_validation_scores_mape.append(history_mape['validation'].iloc[-1])
        
        
        error_test_scores_rmse.append(melhor_arq['melhor arq'].iloc[-1])
        error_test_scores_mape.append(melhor_arq['melhor arq'].iloc[-1])  
        
    
    # revisão Karla
    return error_train_scores_rmse, error_validation_scores_rmse, error_test_scores_rmse

results = DataFrame()


#CONFIGURACOES
# numero de instancias da rede neural para avaliar o processo estocástico da inicializacao das redes neurais
repeats = 10

# numero de registros apresentados a rede para atualização do modelo
# Não ALTERE ESSE PARÂMETRO, POIS ELE É O DIVISOR DO NUMERO DE REGISTROS
# QUE DEFINE O TAMANHO DO CONJUNTO BATCH OU SEJA O TAM_BATCH = NO DE REGISTROS/n_batch

n_batch = 1

#Em séries com sazonalidade anual, mantenha lag =12
lag = [12]

# VARIE O NÚMERO DE ÉPOCAS
#epochs = [400, 500, 600, 700, 800, 900, 1000]
epochs = 100 

# variando o número de neuronios
neurons = [1]#[1, 2, 3, 4, 5, 6, 7] 

for n in neurons:
    # revisao Karla
    #-------------------------------
    e_train,e_val, e_test = run(epochs,n,repeats, n_batch)
    #revisao Karla
    print ('NH = %d train = %f val = %f test = %f' % (n, np.mean(e_train), np.mean(e_val), np.mean(e_test)))
    #-------------------------------

#calcular a media dos erros de validação das 10 instancias
       

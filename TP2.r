library(dplyr)
library(caret)
library(pdp)
library(ICEbox)
library(iml)
library(lime)


#limpiamos la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection



# Establecemos el Working Directory
setwd("C:\\Repos\\") 
# Cargamos el fichero desde CSV 
titanic <- read.csv("train.csv", header = TRUE)
str(titanic)

# Hacemos categorica la variable objetivo
titanic <- titanic %>% mutate(Survived = case_when(Survived == 0 ~ "no",
                                                   Survived == 1 ~ "si"))


# Reservamos cuatro observaciones para que hagan el papel de observaciones nuevas
# a predecir: dos personas se genero femenino y dos personas de genero masculino 
# y una que sobrevivio de cada una (888,889,890, 891)
indices_new <- c(888, 886, 890, 891)
titanic_new <- titanic[indices_new, ] 

# Limpiamos un poco el dataset nuevo
titanic_new$Cabin <- NULL  
titanic_new$PassengerId <- NULL
titanic_new <- titanic_new[!is.na(titanic_new$Age),]
titanic_new$Name <- NULL
titanic_new$Ticket <- NULL

# Creamos el dataset de entrenamiento
titanic_train <- titanic[-indices_new, ]

# Limpiamos un poco el dataset de entranamiento, eliminando nulos
titanic_train$Cabin <- NULL  
titanic_train$PassengerId <- NULL
titanic_train$Name <- NULL
titanic_train$Ticket <- NULL
titanic_train <- titanic_train[!is.na(titanic_train$Age),]


# Procesamiento de los datos
# Método de validación cruzada (5-fold)
fitControl <- trainControl(method = "cv", 
                           number = 3,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary
)

# Ajuste del modelo random forest
set.seed(7)
modelo_rf <- caret::train(Survived ~ ., data = titanic_train, 
                          method = "rf",
                          metric = "ROC",
                          prob.model = TRUE, 
                          trControl = fitControl
)
modelo_rf

# Obtenemos la predicción sobre las observaciones de test
pred_modelo_rf <- data.frame(Survived = predict(object = modelo_rf, 
                                                newdata = titanic_new,
                                                type = "prob"))
pred_modelo_rf



# Unimos las predicciones de la probabilidad de supervivencia a los datos nuevos
titanic_new <- titanic_new %>% cbind(Survived_prob = pred_modelo_rf$Survived.si)
titanic_new


# Función para obtener la probabilidad media de sobrevivencia a partir del modelo RF
prob_fun_pdp <- function(object, newdata){
  mean(predict(object, newdata, type = "prob")[, 2])
}


# PDP (1 predictor)
pdp_titanic <- pdp::partial(modelo_rf, 
                            pred.fun = prob_fun_pdp, 
                            pred.var = "Age", 
                            rug = TRUE)

plotPartial(pdp_titanic, xlab = "Age", ylab = "P(Sobrevive)")


##########################################################################################

# Se muestran la variación de las predicciones en función del valor que toman los predictores
prob_fun_ice <- function(object, newdata){
  predict(object, newdata, type = "prob")[, 2]
}

# Curvas ICE en función de la edad o precio pagado del boleto
ice_titanic <- ice(object = modelo_rf, 
                   X = titanic_train[which(names(titanic_train) != "Survived")],
                   predictor = "Age", 
                   logodds = TRUE,
                   predictfcn = prob_fun_ice,
                   verbose = FALSE)





plot(x = ice_titanic,
     frac_to_plot = 0.7, #50% de observaciones
     plot_orig_pts_preds = TRUE,
     pts_preds_size = 1,
     rug_quantile = seq(from = 0, to = 1, by = 0.01), 
     plot_pdp = TRUE,
     main = "Gráfico ICE: Age",
     xlab = "Age", 
     ylab = "P(Sobrevivir)")

# Mientras que para 10 la edad no pareciera afectar a la supervivenvcia, para valores 
# superioes se observa que si sucede disminuyendo dicha probabilidad.

# Grafico ICE centrado para el predictor "temp" a partir de su valor mínimo
plot(x = ice_titanic,
     frac_to_plot = 0.7, #50% de observaciones
     plot_orig_pts_preds = TRUE,
     pts_preds_size = 1,
     rug_quantile = seq(from = 0, to = 1, by = 0.01), 
     plot_pdp = TRUE,
     centered = TRUE,
     main = "Gráfico c-ICE: Age")

# Compararndo los casos en que el procio pagado por el boleto es menor asi tambien 
# es la probabilidad de sobrevivir y a medida que va a umentando ese valor lo hace 
# tambien dicha probabilidad

dice_titanic <- dice(ice_obj = ice_titanic)

# Gráfico ICE derivado
plot(x = dice_titanic,
     frac_to_plot = 0.7,
     plot_orig_pts_deriv = TRUE,
     pts_preds_size = 1,
     rug_quantile = seq(from = 0, to = 1, by = 0.01),
     plot_dpdp = TRUE,
     main = "Gráfico d-ICE: Age")

# Se olbervan que las derivadas parciales no son
# paralelas indicando que hay interaccion entre los predictores.


###############################################
########### INTERACCIONES #####################
###############################################


predictor <- Predictor$new(modelo_rf, 
                           data = titanic_train[which(names(titanic_train) != "Survived")],
                           y = titanic_train$Survived)

plot(iml::Interaction$new(predictor))


###################
#LIME#
###################



# Creamos el objeto lime "explicador"
explainer_lime <- lime(x = titanic_train[which(names(titanic_train) != "Survived")], 
                       model = modelo_rf,
                       bin_continuous = TRUE,
                       quantile_bins = FALSE)

summary(explainer_lime)





# Obtenemos la explicación sobre las observaciones de test
titanic_new1 <- titanic_new
titanic_new1$Survived <- NULL
titanic_new1$Survived_prob <- NULL

remove.packages("lime")
install.packages("lime")
library(lime)

lime_titanic <- explain(x = titanic_new1, 
                        explainer = explainer_lime,
                        n_labels = 1,
                        n_features = 8) 


tibble::glimpse(lime_titanic)






# Visualización del resultado
plot_features(lime_titanic)

# Trazará las 8 variables más influyentes que mejor explican el modelo lineal en
# esa región local de observaciones y si la variable provoca un aumento en la 
# probabilidad (respalda) o una disminución en la probabilidad (contradice). 
# También nos proporciona el ajuste del modelo para cada modelo ("Explanation fit: XX"),
# lo que nos permite ver qué tan bien ese modelo explica la región local.

# En consecuencia, podemos inferir que el caso 888 tiene la mayor probabilidad de 
# supervivencia de las 4 observaciones y las variables que parecen estar influyendo 
# en esta alta probabilidad incluyen Sex, Age y Pclass

# con lo cual podriamos ver que el modelo tomo como importante que una mujer de unos 20 años de las primeras clases 
# tuvo mayores probabilidades de sobrevivir.



plot_explanations(lime_titanic)

# El otro gráfico que podemos crear es un mapa de calor que muestra cómo las 
# diferentes variables seleccionadas en todas las observaciones influyen en cada caso. 
# Esta gráfica se vuelve útil si está tratando de encontrar características comunes 
# que influyan en todas las observaciones o si está realizando este análisis 
# en muchas observaciones que plot_featuresdificultan el discernimiento.

explainer_lime1 <- lime(x = titanic_train[which(names(titanic_train) != "Survived")], 
                       model = modelo_rf,
                       bin_continuous = TRUE,
                       quantile_bins = FALSE)

summary(explainer_lime)

lime_titanic1 <- explain(x = titanic_train[which(names(titanic_train) != "Survived")], 
                        explainer = explainer_lime1,
                        n_labels = 1,
                        n_features = 8) 


plot_explanations(lime_titanic1)

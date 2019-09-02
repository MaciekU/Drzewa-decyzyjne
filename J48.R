
library(RWeka)





# iris -----------------------------------------------------------------


data(iris)
## Learn J4.8 tree with reduced error pruning (-R) and
## minimum number of instances set to 5 (-M 5):
##J48(Species ~ ., data = iris, control = Weka_control(R = TRUE, M = 5))

## opcje WOW("J48")
## U unpruned tree
## B używaj tylko binary splits
## C confidenceFactor The confidence factor used for pruning (smaller values incur more pruning) (default 0.25)
##pruning zapobiega overfitting (gorsze wyniki dla testowych ale drzewo jest bardziej ogólne i nie wyspecyfikowane na drzewo treningowe)
## M minimalna liczba instancji na liść (default 2)
## N Amount of data used for reduced-error pruning. One fold is used for pruning, the rest for growing the tree. (default 3)
## -num-decimal-places The number of decimal places for the output of numbers in the model (default 2).
## -batch-size The desired batch size for batch prediction (default 100).
fit <- J48(Species ~ ., data = iris, control = Weka_control(R = TRUE  ) )
summary(fit)
predictions <- predict(fit, iris[,1:4])
table(predictions, iris$Species)
##uruchom bnlearn 
#podzial = kmeans(iris[,1:4],10)##podzial na 10 zbiorów

library("Rgraphviz")
ff <- tempfile()
write_to_dot(fit, ff)
plot(agread(ff))
##if(require("party", quietly = TRUE)) plot(fit)

# glass --------------------------------------------------------------------


glass = read.csv("programy/glass.data", header = FALSE, sep = ",")
names(glass) <- c("Id number","RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type_of_glass")
glass = glass[-1] ##usuwam pierwszą kolumnę ID
glass$`Type_of_glass` <- factor(glass$`Type_of_glass`)
fit2 <- J48(Type_of_glass ~ .,data = glass, control = Weka_control(R = TRUE))
res <- evaluate_Weka_classifier(fit2,numFolds=10, class=TRUE)
print(res)
summary(fit2)
predictions <- predict(fit2, glass[,1:9])
table<-table(predictions,glass$Type_of_glass)
library("Rgraphviz")
ff2 <- tempfile()
write_to_dot(fit2, ff2)
#plot(g1, nodeAttrs=makeNodeAttrs(g1, fontsize=4))
plot(agread(ff2))


# diabetes -------------------------------------------------------------------



diabetes = read.table("programy/pima_indians_diabetes.txt", header = FALSE, sep = ",")
names(diabetes) <- c("No_pregnant", "Plasma_glucose", "Blood_pres", "Skin_thick", "Serum_insu", "BMI", "Diabetes_func", "Age", "Class")
diabetes$`Class` <- factor(diabetes$`Class`)
fit3 <- J48(Class ~ .,data = diabetes, control = Weka_control(R = TRUE))
res <- evaluate_Weka_classifier(fit3,numFolds=10, class=TRUE)
print(res)
summary(fit3)
predictions <- predict(fit3, diabetes[,1:8])
table(predictions,diabetes$Class)
library("Rgraphviz")
ff3 <- tempfile()
write_to_dot(fit3, ff3)
plot(agread(ff3))

# wine ----------------------------------------------------------------


wine = read.table("programy/wine.data", header = FALSE, sep = ",")
names(wine) <- c("Class","Alcohol","Malic Acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline")
wine$`Class` <- factor(wine$`Class`)
fit4 <- J48(Class ~ .,data = wine, control = Weka_control(R = FALSE, M=4))
res <- evaluate_Weka_classifier(fit4,numFolds=10, class=TRUE)
print(res)
summary(fit4)
predictions <- predict(fit4, wine[,2:14])
table(predictions,wine$Class)
library("Rgraphviz")
ff4 <- tempfile()
write_to_dot(fit4, ff4)
plot(agread(ff4))

# rest --------------------------------------------------------------------

#defAttrs <- getDefaultAttrs()
#defAttrs$node$fontsize <- character(30)

##subs <- c(sample()) ##vector
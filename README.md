# System Rozpoznawania Podpisu
Simple signature recognition system

### Projekt powinien zawierać: 
- bazę danych własną (lub  jedną z udostępnioną w internecie) 

- opis użytych algorytmów i utworzonego programu (własnego, z użyciem dostępnych bibliotek)

- opis wykonanych eksperymentów  

TODO: wymyślić proste eksperymenty - pomysły: Sampled Hyperparameter Tuning (czyli na subsecie danych), wytrenowanie gotowego modelu i porównanie z naszym (np. VGG16, ResNet)

- wnioski 

## Baza danych

Do wytrenowania sieci skorzystano z bazy danych znajdującej się na kaggle`u:
https://www.kaggle.com/datasets/shreelakshmigp/cedardataset/data

W zbiorze danych znajdują się dwa foldery:
 - full_forg 
 - full_org 

Każdy z nich posiada po 1320 zdjęć podpisów o różnych rozmiarach w kształcie prostokąta.
Na te 1320 zdjęć składają się po 24 podpisy, 55 różnych autorów.
Pierwszy folder (full_forg) zawiera w sobie podrobione podpisy natomiast drugi (full_org) to oryginalne zdjęcia podpisów.
Rozmiary zdjęć z podpisami wahają się dla długości między 150 a 810 pikseli, a dla szerokości między 270 a 890 piskeli.

Przy testowaniu wykorzystano również bazę danych z kaggle:
https://www.kaggle.com/datasets/adeelajmal/gpds-1150

W tym przypadku zbiór danych składa się z dwóch podfolderów:
 - test
 - train

Pierwszy charakteryzuje się 150 kolejnymi podfolderami. Każdy z nich posiada dwa foldery. Pierwszy z tych folderów to podrobione podpisy w ilości 14. Drugi to 8 zdjęć prawdziwych podpisów.
Drugi to zbiorczy folder w którym dane podzielone są na foldery, fałszywe i prawdziwe podpisy. Obydwa posiadają po 2400 zdjęć.

## Opis algorytmów i programu

Algorytm analizy podpisów wykorzystujący sieć neuronową CedarNetwork opiera się na konwolucyjnej sieci neuronowej (CNN), która jest specjalnie zaprojektowana do przetwarzania i klasyfikacji obrazów podpisów jako prawdziwych lub fałszywych. 

### Architektura sieci

Warstwy konwolucyjne i pooling:
Sieć składa się z trzech warstw konwolucyjnych, które są połączone z warstwami pooling. Warstwy konwolucyjne służą do wykrywania różnych cech obrazu, takich jak krawędzie, tekstury i wzory charakterystyczne dla podpisów.
Warstwy pooling (max pooling) zmniejszają rozmiar danych, co pomaga w redukcji liczby parametrów i zapobiega przeuczeniu modelu.

Warstwy w pełni połączone:
Po przejściu przez warstwy konwolucyjne, dane są spłaszczane i przekazywane do dwóch w pełni połączonych warstw. Te warstwy działają jak klasyfikator, który na podstawie wyekstrahowanych cech podejmuje decyzję o przynależności obrazu do jednej z dwóch klas: prawdziwy lub fałszywy podpis.
Funkcja aktywacji Sigmoid:

Ostateczna warstwa sieci używa funkcji aktywacji Sigmoid, która przekształca wyjście sieci w wartość z przedziału [0, 1]. Dzięki temu sieć może interpretować wynik jako prawdopodobieństwo przynależności do klasy "prawdziwy podpis".

### Proces treningu

Proces treningu sieci CedarNetwork obejmuje kilka kluczowych etapów:

Przygotowanie zbioru danych:
Algorytm wykorzystuje zbiór danych zawierający obrazy prawdziwych i fałszywych podpisów. Dane są podzielone na zestaw treningowy i walidacyjny.

Inicjalizacja modelu:
Na początku procesu treningu tworzymy instancję sieci CedarNetwork oraz definiujemy funkcję straty (Binary Cross-Entropy Loss) i optymalizator (Adam).

Pętla treningowa:
Model jest trenowany przez kilka epok, w każdej epoce przechodząc przez wszystkie próbki w zestawie treningowym. Dla każdej próbki obliczana jest strata, a wagi modelu są aktualizowane w celu minimalizacji tej straty za pomocą propagacji wstecznej.

Walidacja modelu:
Po każdej epoce model jest oceniany na zestawie walidacyjnym. Obliczana jest strata walidacyjna oraz dokładność modelu. Na tej podstawie monitorowane jest działanie sieci, co pozwala na wczesne zatrzymanie treningu, jeśli dokładność osiągnie odpowiedni poziom lub jeśli nie ma poprawy przez określoną liczbę epok.

## Eksperymenty


CEDAR
| Epoch | Loss                 | Val Loss              | Accuracy           | Precision          | Recall             | F1 Score           | Elapsed Time       |
|-------|----------------------|-----------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| 1     | 0.7459750753460508   | 0.6884136690812952    | 0.9223484848484849 | 0.9916666666666667 | 0.8592057761732852 | 0.9206963249516441 | 28.606995820999146 |
| 2     | 0.6796087731014598   | 0.6329151356921476    | 0.6117424242424242 | 0.5967741935483871 | 0.8014440433212996 | 0.6841294298921418 | 28.889320373535156 |
| 3     | 0.5971536180286696   | 0.5214245301835677    | 0.7234848484848485 | 0.7027863777089783 | 0.8194945848375451 | 0.7566666666666667 | 28.399161100387573 |
| 4     | 0.618424812501127    | 0.4859549631090725    | 0.8522727272727273 | 0.9853658536585366 | 0.7292418772563177 | 0.8381742738589212 | 28.30045247077942  |
| 5     | 0.26383901658383285  | 0.03750317565658513   | 0.9962121212121212 | 1.0                | 0.9927797833935018 | 0.9963768115942029 | 28.669886827468872 |
| 6     | 0.16031875482065405  | 0.017358707537984148  | 1.0                | 1.0                | 1.0                | 1.0                | 28.93173837661743  |
| 7     | 0.004495310996196967 | 0.0007762849683572046 | 1.0                | 1.0                | 1.0                | 1.0                | 28.524136543273926 |

RESNET
| Epoch | Loss                 | Val Loss               | Accuracy           | Precision          | Recall             | F1 Score           | Elapsed Time       |
|-------|----------------------|------------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| 1     | 0.045614429219505946 | 0.6690967416062075     | 0.7916666666666666 | 0.7215189873417721 | 1.0                | 0.8382352941176471 | 47.574694871902466 |
| 2     | 0.007402830922342908 | 0.06789086359169554    | 0.9696969696969697 | 1.0                | 0.9438596491228071 | 0.9711191335740073 | 48.737656593322754 |
| 3     | 0.00424626990460445  | 0.007048115004127955   | 0.9981060606060606 | 1.0                | 0.9964912280701754 | 0.9982425307557118 | 49.06009793281555  |
| 4     | 0.000994741135262743 | 0.00013966734075117581 | 1.0                | 1.0                | 0.0                | 1.0                | 47.76256608963013  |


VGG16
| Epoch | Loss               | Val Loss           | Accuracy | Precision | Recall | F1 Score           | Elapsed Time       |
|-------|--------------------|--------------------|----------|-----------|--------|--------------------|--------------------|
| 1     | 0.7993745903174082 | 0.6956436002955717 | 0.5      | 0.0       | 0.0    | 0.0                | 319.53003191947937 |
| 2     | 0.6973494535142725 | 0.6996395763228921 | 0.5      | 0.5       | 1.0    | 0.6666666666666666 | 358.5489385128021  |
| 3     | 0.6935828716465922 | 0.698022632037892  | 0.5      | 0.5       | 1.0    | 0.6666666666666666 | 351.940541267395   |
| 4     | 0.695623517036438  | 0.6984934982131509 | 0.5      | 0.0       | 0.0    | 0.0                | 301.6934518814087  |

## Wnioski

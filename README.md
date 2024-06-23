# System Rozpoznawania Podpisu
Simple signature recognition system

### Projekt powinien zawierać: 
- bazę danych własną (lub  jedną z udostępnioną w internecie) 

Poza CEDARem możemy użyć GPDS do samego testowania pojedynczych podpisów

https://www.kaggle.com/datasets/adeelajmal/gpds-1150
- opis użytych algorytmów i utworzonego programu (własnego, z użyciem dostępnych bibliotek) 

Trzeba dodać opis modelu

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

## Wnioski

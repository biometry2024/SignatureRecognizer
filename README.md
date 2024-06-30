# SignatureRecognizer - System Rozpoznawania Podpisu

Autorzy: Jakub Bochenek, Adam Walka, Konrad Micek

## Baza danych

Do wytrenowania sieci skorzystano z bazy danych znajdującej się na kaggle`u:
https://www.kaggle.com/datasets/shreelakshmigp/cedardataset/data

W zbiorze danych znajdują się dwa foldery:
 - full_forg - sfałszowane podpisy,
 - full_org - prawdziwe podpisy.

Każdy z nich posiada po 1320 zdjęć podpisów o różnych rozmiarach w kształcie prostokąta.
Na te 1320 zdjęć składają się po 24 podpisy, 55 różnych autorów.
Pierwszy folder (full_forg) zawiera w sobie podrobione podpisy natomiast drugi (full_org) to oryginalne zdjęcia podpisów.
Rozmiary zdjęć z podpisami wahają się dla długości między 150 a 810 pikseli, a dla szerokości między 270 a 890 piskeli.


## Opis algorytmów i programu

Algorytm analizy podpisów wykorzystujący sieć neuronową CedarNetwork opiera się na konwolucyjnej sieci neuronowej (CNN), która jest specjalnie zaprojektowana do przetwarzania i klasyfikacji obrazów podpisów jako prawdziwych lub fałszywych. 

### Architektura sieci

#### Warstwy konwolucyjne i pooling

Sieć składa się z trzech warstw konwolucyjnych, które są połączone z warstwami pooling. Warstwy konwolucyjne służą do wykrywania różnych cech obrazu, takich jak krawędzie, tekstury i wzory charakterystyczne dla podpisów.
Warstwy pooling (max pooling) zmniejszają rozmiar danych, co pomaga w redukcji liczby parametrów i zapobiega przeuczeniu modelu.

#### Warstwy w pełni połączone

Po przejściu przez warstwy konwolucyjne, dane są spłaszczane i przekazywane do dwóch w pełni połączonych warstw. Te warstwy działają jak klasyfikator, który na podstawie wyekstrahowanych cech podejmuje decyzję o przynależności obrazu do jednej z dwóch klas: prawdziwy lub fałszywy podpis.

#### Funkcja aktywacji Sigmoid

Ostateczna warstwa sieci używa funkcji aktywacji Sigmoid, która przekształca wyjście sieci w wartość z przedziału [0, 1]. Dzięki temu sieć może interpretować wynik jako prawdopodobieństwo przynależności do klasy "prawdziwy podpis".

### Proces treningu

Proces treningu sieci CedarNetwork obejmuje kilka kluczowych etapów:

Przygotowanie zbioru danych:
Algorytm wykorzystuje zbiór danych zawierający obrazy prawdziwych i fałszywych podpisów. Dane są podzielone na zestaw treningowy i walidacyjny.

Inicjalizacja modelu:
Na początku procesu treningu tworzymy instancję sieci CedarNetwork oraz definiujemy funkcję straty (Binary Cross-Entropy Loss) i optymalizator (Adam).

Pętla treningowa:
Model jest trenowany przez kilka epok, w każdej epoce przechodząc przez wszystkie próbki w zestawie treningowym. Dla każdej próbki obliczana jest strata, a wagi modelu są aktualizowane w celu minimalizacji tej straty za pomocą propagacji wstecznej. Zasotsowany zostął mechanizm wczesnego zakańczania treningu poprzez porównywanie celności oraz strat modelu w kolejnych epokach.

Walidacja modelu:
Po każdej epoce model jest oceniany na zestawie walidacyjnym. Obliczana jest strata walidacyjna oraz dokładność modelu. Na tej podstawie monitorowane jest działanie sieci, co pozwala na wczesne zatrzymanie treningu, jeśli dokładność osiągnie odpowiedni poziom lub jeśli nie ma poprawy przez określoną liczbę epok.

### Funkcje programu

Program pozwala na:
- sprawdzenie oryginalności pojedynczego podpisu poprzez podanie pełnej ścieżki do zdjęcia,
- wytrenowanie modelu na nowo,
- dotrenowanie wytrenowanego już modelu.

## Eksperymenty

### Porównanie między różnymi sieciami

Do porównania wykorzystano własny stworzony model - Cedar oraz dwa inne popularne modele znajudjące się w bibliotece PyTorch: ResNet i VGG16.
Parametry znajdujące się w tabelach:
 - Epoch - Numer epoki treningowej. Jest to jedno pełne przejście przez cały zbiór treningowy.
 - Loss - Wartość funkcji straty na zbiorze treningowym. Mniejsza wartość oznacza lepsze dopasowanie do danych treningowych.
 - Val Loss - Wartość funkcji straty na zbiorze walidacyjnym. Miara jak dobrze model generalizuje do nowych, niewidzanych danych.
 - Accuracy - Precyzja modelu. Odsetek prawdziwie pozytywnych przewidywań spośród wszystkich pozytywnych przewidywań.
 - Recall - Czułość modelu, czyli odsetek prawdziwie pozytywnych przewidywań spośród wszystkich rzeczywistych pozytywnych przypadków.
 - F1 Score - Średnia harmoniczna precyzji i czułości, używana do oceny modelu, zwłaszcza przy niezrównoważonych klasach.
 - Elapsed Time - Czas, jaki zajęła jedna epoka treningowa (w sekundach).

Wszystkie testy były wykonywane na CPU w wyniku braku dostępności GPU - wykorzystując GPU można by się spodziewać co najmniej kilkukrotnego skrócenia czasu treningu. Każdy model był trenowany na jednakowych parametrach. Poniżej znajdują się tabele wraz z wartościami przy trenowaniu sieci. 

#### Cedar
| Epoch | Loss                 | Val Loss              | Accuracy           | Precision          | Recall             | F1 Score           | Elapsed Time       |
|-------|----------------------|-----------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| 1     | 0.7459750753460508   | 0.6884136690812952    | 0.9223484848484849 | 0.9916666666666667 | 0.8592057761732852 | 0.9206963249516441 | 28.606995820999146 |
| 2     | 0.6796087731014598   | 0.6329151356921476    | 0.6117424242424242 | 0.5967741935483871 | 0.8014440433212996 | 0.6841294298921418 | 28.889320373535156 |
| 3     | 0.5971536180286696   | 0.5214245301835677    | 0.7234848484848485 | 0.7027863777089783 | 0.8194945848375451 | 0.7566666666666667 | 28.399161100387573 |
| 4     | 0.618424812501127    | 0.4859549631090725    | 0.8522727272727273 | 0.9853658536585366 | 0.7292418772563177 | 0.8381742738589212 | 28.30045247077942  |
| 5     | 0.26383901658383285  | 0.03750317565658513   | 0.9962121212121212 | 1.0                | 0.9927797833935018 | 0.9963768115942029 | 28.669886827468872 |
| 6     | 0.16031875482065405  | 0.017358707537984148  | 1.0                | 1.0                | 1.0                | 1.0                | 28.93173837661743  |
| 7     | 0.004495310996196967 | 0.0007762849683572046 | 1.0                | 1.0                | 1.0                | 1.0                | 28.524136543273926 |

#### ResNet18
| Epoch | Loss                 | Val Loss               | Accuracy           | Precision          | Recall             | F1 Score           | Elapsed Time       |
|-------|----------------------|------------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| 1     | 0.045614429219505946 | 0.6690967416062075     | 0.7916666666666666 | 0.7215189873417721 | 1.0                | 0.8382352941176471 | 47.574694871902466 |
| 2     | 0.007402830922342908 | 0.06789086359169554    | 0.9696969696969697 | 1.0                | 0.9438596491228071 | 0.9711191335740073 | 48.737656593322754 |
| 3     | 0.00424626990460445  | 0.007048115004127955   | 0.9981060606060606 | 1.0                | 0.9964912280701754 | 0.9982425307557118 | 49.06009793281555  |
| 4     | 0.000994741135262743 | 0.00013966734075117581 | 1.0                | 1.0                | 0.0                | 1.0                | 47.76256608963013  |


#### VGG16
| Epoch | Loss               | Val Loss           | Accuracy | Precision | Recall | F1 Score           | Elapsed Time       |
|-------|--------------------|--------------------|----------|-----------|--------|--------------------|--------------------|
| 1     | 0.7993745903174082 | 0.6956436002955717 | 0.5      | 0.0       | 0.0    | 0.0                | 319.53003191947937 |
| 2     | 0.6973494535142725 | 0.6996395763228921 | 0.5      | 0.5       | 1.0    | 0.6666666666666666 | 358.5489385128021  |
| 3     | 0.6935828716465922 | 0.698022632037892  | 0.5      | 0.5       | 1.0    | 0.6666666666666666 | 351.940541267395   |
| 4     | 0.695623517036438  | 0.6984934982131509 | 0.5      | 0.0       | 0.0    | 0.0                | 301.6934518814087  |

#### Wnioski

VGG16 zupełnie nie poradził sobie z zestawem Cedar. Przez 4 iteracje zbioru danych jego wydajność nie zmieniała się, do tego każdy okres trwał bardzo długo w porównaniu z innymi modelami.

ResNet18 uzyskał bardzo dobre wyniki. Osiągnął świetną celność już po 4 epokach, a ich czas trwania był dosyć krótki.

Autorski model ze względu na mniejszą ilość warstw w porównaniu z pozostałymi modelami miał najkrótszy czas epoki, jednak do osiągnięcia bardzo dobrych wyników potrzebował 5-6 epok. Trzeba jednak przyznać, że pomimo swojej prostoty model oferuje bardzo dobre parametry i dla wybranego zbioru danych jest konkurencyjny z modelem ResNet.

### Kilkukrotne przetrenowanie autorskiej sieci

Test był wykonywany na innym, słabszym sprzętowo komputerze niż powyższe testy, stąd znacznie dłuższy czas ich wykonania.

#### Trening 1

| Epoch | Loss                 | Val Loss              | Accuracy           | Precision          | Recall             | F1 Score           | Elapsed Time       |
|-------|----------------------|-----------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| 1     | 0.7294277593945012   | 0.6915848079849692    | 0.5018939393939394 | 0.5018939393939394 | 1.0                | 0.6683480453972258 | 110.62269949913025 |
| 2     | 0.6918990846836206   | 0.693884551525116     | 0.4981060606060606 | 0.0                | 0.0                | 0.0                | 102.78726029396057 |
| 3     | 0.6105143091443813   | 0.4341328056419597    | 0.75               | 0.6675062972292192 | 1.0                | 0.8006042296072508 | 113.45713138580322 |
| 4     | 0.2847223316008846   | 0.005021942868445288  | 1.0                | 1.0                | 1.0                | 1.0                | 123.36775755882263 |
| 5     | 0.3310140148757703   | 0.013874418923960012  | 1.0                | 1.0                | 1.0                | 1.0                | 126.78728365898132 |
| 6     | 0.006319532081816402 | 0.0016084651544909267 | 1.0                | 1.0                | 1.0                | 1.0                | 142.45789504051208 |


#### Trening 2

| Epoch | Loss                 | Val Loss              | Accuracy           | Precision          | Recall             | F1 Score           | Elapsed Time       |
|-------|----------------------|-----------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| 1     | 0.7115344459360297   | 0.6769058318699107    | 0.5037878787878788 | 0.0                | 0.0                | 0.0                | 145.56020331382751 |
| 2     | 0.6711439824465549   | 0.5698645500575795    | 0.4962121212121212 | 0.4962121212121212 | 1.0                | 0.6632911392405064 | 111.97881960868835 |
| 3     | 0.20190343504884478  | 0.048907082747010625  | 0.9867424242424242 | 0.9739776951672863 | 1.0                | 0.9868173258003766 | 118.71889305114746 |
| 4     | 0.23747497914838744  | 0.16690007641034968   | 0.9375             | 0.888135593220339  | 1.0                | 0.940754039497307  | 127.79018807411194 |
| 5     | 0.026731101640810568 | 0.0108984291608281    | 1.0                | 1.0                | 1.0                | 1.0                | 137.84848761558533 |
| 6     | 0.010570581191021836 | 0.0036667491761310134 | 1.0                | 1.0                | 1.0                | 1.0                | 113.51597213745117 |


#### Trening 3

| Epoch | Loss                 | Val Loss              | Accuracy           | Precision          | Recall             | F1 Score           | Elapsed Time       |
|-------|----------------------|-----------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| 1     | 0.7093636890252432   | 0.6814743070041432    | 0.5056818181818182 | 0.0                | 0.0                | 0.0                | 134.0037121772766  |
| 2     | 0.5835249392372189   | 0.39906858083079844   | 0.7973484848484849 | 0.7092391304347826 | 1.0                | 0.8298887122416534 | 120.74614024162292 |
| 3     | 0.3004613954037654   | 0.01017524371854961   | 1.0                | 1.0                | 1.0                | 1.0                | 145.8005096912384  |
| 4     | 0.003760965242470389 | 0.002594727485676926  | 1.0                | 1.0                | 1.0                | 1.0                | 151.7447738647461  |
| 5     | 0.21126053923261212  | 0.011324645403553458  | 1.0                | 1.0                | 1.0                | 1.0                | 137.94452929496765 |


#### Wnioski

Początkowy słaby start:

W każdej serii treningowej model zaczyna z niską dokładnością (ok. 50%) i wysokimi wartościami Loss i Val Loss w pierwszej epoce.
Brak precyzji i wartości Recall na poziomie 0 w pierwszej epoce w niektórych przypadkach wskazuje na problemy z klasyfikacją na starcie.

Znacząca poprawa w kolejnych epokach:

W kolejnych epokach model szybko poprawia swoje wyniki, osiągając wysoką dokładność, precyzję, Recall i F1 Score.
Już w 3. epoce wartości te są znacznie lepsze, a Val Loss mocno spada, wskazując na poprawę klasyfikacji.

Stabilna wysoka wydajność:

Po 4. epoce model osiąga maksymalną dokładność, precyzję, Recall i F1 Score wynoszącą 1.0, co wskazuje na idealną klasyfikację.
Loss i Val Loss również spadają do bardzo niskich wartości, co wskazuje na dobrą generalizację modelu.

Czas treningu:

Czas treningu rośnie wraz z każdą epoką, co może być związane ze zwiększoną ilością danych do przetworzenia i bardziej skomplikowanymi obliczeniami.

Model wykazuje zdolność szybkiego uczenia się i osiągania wysokiej wydajności już w pierwszych kilku epokach. Początkowe trudności z klasyfikacją są szybko pokonywane, a model stabilnie utrzymuje wysoką dokładność i jakość klasyfikacji. Jest to pozytywny sygnał, sugerujący, że model pomimo swojej prostoty jest dobrze dostosowany do danych i ma zdolność do generalizacji.

## Podsumowanie

Stworzony program będący siecią neuronową służącą do rozpoznawania oryginalności podpisu działa i daje świetne wyniki dla podpisów z podanej bazy pomimo prostoty budowy sieci. Sama sieć nie jest duża, co również można powiedzieć o zbiorze treningowym. Sam zbiór to tylko 55 różnych autorów, a sieć składa się z czterech warstw. 

Jednakże porównując sieć do innych, gotowych modeli można zauważyć, że autorska sieć jest szybsza, a mimo to osiąga wysoką dokładność i precyzję. Rozbudowjąc sieć i gromadząc większy zbiór danych, istnieje możliwość stworzenia dokładnego i szybkiego modelu rozpoznawania oryginalności podpisów.

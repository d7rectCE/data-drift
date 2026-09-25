# Связанные работы: где driftfdr стоит относительно литературы

[English](related_work.md) · **Русский**

Черновик раздела для статьи. Номера в квадратных скобках — источники из обзора литературы
(«Контроль ложных тревог при мониторинге дрейфа в системах с множеством ML-моделей»; полный
список — в [конце документа](#список-литературы)); там же
таблица соответствия постановок трёх направлений. Здесь к каждому направлению добавлено, что
показали эксперименты driftfdr (номера экспериментов — по [журналу](experiments.ru.md)).
Работы, которых нет в обзоре, помечены отдельно; их детали нужно сверить по оригиналам
перед отправкой статьи.

## Детекторы дрейфа в ML [1–8]

ADWIN, DDM, Page-Hinkley и KS выдают бинарный сигнал с неизвестной фактической частотой
ложных тревог, тем более на автокоррелированном потоке.

- **Что делает driftfdr.** Каждый детектор рассматривается как непрерывная статистика с одним
  параметром чувствительности. Её нулевое распределение оценивается бутстрепом с учётом
  автокорреляции, а сигнал детектора превращается в p-значение (эксп. 1). Для Page-Hinkley и DDM
  статистика совпадает с river.
- **Что показали эксперименты.** При автокорреляции обычный бутстреп теряет контроль: ложных
  тревог в разы больше заявленного; блочный антиконсервативен в 1.5–3.5 раза, AR-sieve ближе всего
  к номиналу (эксп. 1). С учётом неопределённости параметров и GPD-хвостом уровень выдерживается
  для PH, KS и MeanShift, но не для ADWIN (эксп. 10, 15). Детекторы на признаках ловят не тот дрейф: они
  тревожат на безвредном сдвиге p(X) и не видят вредного сдвига p(y|X) (эксп. 11).

## Бенчмарки детекторов [2, 8]

Cerqueira и др. [2] сравнивают детекторы по F1 после настройки, но при разном фактическом
уровне ложных тревог; потоки перемешаны, поэтому автокорреляции нет.

- **Что делает driftfdr.** Все детекторы сравниваются при одном и том же фактическом уровне
  ложных тревог, на автокоррелированных потоках (эксп. 15), и на фиксированном наборе из 100
  сценариев (`benchmark_suite`, эксп. 21). В `summarize` есть та же событийная точность,
  полнота и F1, что и у [2], так что числа сопоставимы.
- **Что показали эксперименты.** Page-Hinkley, один из худших в [2], при едином уровне ложных
  тревог сильнее KS и DDM и к третьему окну не уступает ADWIN; самый мощный — простой сдвиг
  среднего (MeanShift) (эксп. 15). Прямо с [2] это не сравнимо (другие потоки и набор
  детекторов), но показывает, что ранжирование детекторов зависит от того, выровнен ли уровень
  ложных тревог, — довод обзора против сравнения при разных уровнях.
- **Пороги по умолчанию не переносятся между задачами.** Page-Hinkley river по умолчанию на
  ошибках 0/1 даёт 26–75% ложных переобучений (эксп. 14) и F1 0.06 на бенчмарке (эксп. 21), а
  на дневной потере моделей волатильности за 14 лет не тревожит ни разу (эксп. 22): порог λ = 50
  задан в единицах сигнала. Калиброванный порог от единиц сигнала не зависит.

## Мониторинг многих моделей на практике [9, 10]

Rombouts и Wilms тестируют каждый из 32 районов на уровне 5% без поправки, причём фактический
уровень теста 6.7–7.4%, а корреляция спроса между районами 77%.

- **Что показали эксперименты.** Без поправки ложные тревоги растут линейно с числом моделей;
  Бонферрони в окне держит уровень при любой корреляции (эксп. 2, 13). Ожидание обзора, что
  ложные тревоги при корреляции придут пачками, подтвердилось: с BH в одном окне их до
  полутора десятков (эксп. 18), при сильной корреляции — десятки (эксп. 5). На реальных наборах корреляция p-значений между моделями 0.4–0.9
  (эксп. 20). Выделение общего компонента (`split_common`) снимает большую её часть и
  убирает пачки; общий компонент проверяется отдельно как событие всего парка (эксп. 18, 20).

## Разладка в параллельных потоках [11–14]

Процедуры Mei, Chen–Zhang–Poor и Chen–Li строят свои статистики по самим наблюдениям потоков
и не работают с сигналами готовых детекторов. Dandapanthula и Ramdas [14] показали, что при
конечном ARL худший случай классических FDR и FWER тривиален, и ввели ошибку на терпение (EOP).

- **Связь с driftfdr.** driftfdr не заменяет эти процедуры, а подключает к той же логике
  детекторы из river. Мера ошибки, которую мы в итоге рекомендуем, — вероятность хотя бы одной
  ложной тревоги в окне (Бонферрони в окне, эксп. 13) — это ошибка за единицу времени, а не
  за весь прогон, и по духу близка к EOP: она не вырождается с ростом длины мониторинга.
  Формальное сопоставление с EOP — открытый вопрос для статьи.
- **e-детекторы в driftfdr.** `ECUSUM` — смешанный CUSUM e-детектор (Shin, Ramdas & Rinaldo) на
  отбелённых AR-моделью остатках, а `StreamingMonitor(sequential=True)` проверяет его на каждом
  шаге с правилом бонферрониевского типа по моделям, в духе процедуры e-d-Bonferroni из [14]. Два
  отличия: порог калибруется бутстрепом под бюджет ложных тревог на окно, а не выводится из
  e-значений (остатки не точно гауссовы, AR-модель оценена), и гарантия — вероятность ложной
  тревоги в окне, а не EOP. Эксперимент 23 сравнивает его с оконными детекторами.
- Доля ложных за весь прогон (`fdp`) ближе к постановке Chen–Zhang–Poor [12]. BH в окне её
  не контролирует при коррелированных моделях (эксп. 13).

## Онлайн-контроль множественных сравнений [15–21]

Alpha-investing, LORD, SAFFRON требуют валидных p-значений; Rebjock и др. [21] применили
онлайн-FDR к скорам аномальности во временных рядах.

- **Что показали эксперименты.** Когда статистика накапливает свидетельство по нескольким
  окнам, онлайн-процедуры (LORD++, SAFFRON, LOND) проигрывают по задержке при равном числе
  ложных тревог, потому что тратят бюджет ошибки неравномерно во времени (эксп. 3, 12).
  BH внутри окна окупается, когда дрейфы приходят кластерами (эксп. 4); e-BH с гарантией при
  любой зависимости слишком осторожен: треть дрейфов пропущена (эксп. 13).
- **Отличие от [21].** Мы калибруем не скоры, а статистики детекторов дрейфа на ошибке модели,
  и отдельно разбираем сдвиг p(X) и p(y|X) (эксп. 11).

## Калибровка при зависимых данных [22–24]

Блочный [22] и стационарный [23] бутстреп сохраняют зависимость; Wu и Apley [24] показали,
что вложенный бутстреп недооценивает изменчивость статистики, и предложили поправку для одной
MEWMA-карты.

- **Что делает driftfdr.** Блочный, стационарный и AR-sieve бутстреп для семейства детекторов;
  неопределённость оценки параметров учитывается повторной оценкой AR-модели на каждой реплике
  (`sieve_pu`) — та же проблема, что у [24], решённая иначе; поправка 0.632 из [24] не
  реализована. Глубокий хвост, нужный при поправке на тысячи моделей, экстраполируется
  GPD-распределением (эксп. 6, 10).

## Вне обзора (сверить по оригиналам)

- **Podkopaev и Ramdas, «Tracking the risk of a deployed model and detecting harmful
  distribution shifts» (ICLR 2022).** Последовательный тест того, что риск модели вырос больше
  заданного допуска, с гарантией в любой момент времени. Это та же идея, что наша нулевая
  гипотеза «существенного ухудшения» с допуском δ (эксп. 8): тревожить не на любой сдвиг, а на
  вредный. Отличия: у них одна модель и свой последовательный тест, у нас — калибровка
  существующих детекторов, парк моделей и поправка на множественность. В статье нужно
  сослаться как на ближайшую работу по постановке нуля.
- **CBPE в NannyML (оценка качества без меток по предсказанным вероятностям).** Оценивает
  точность модели до прихода меток, предполагая откалиброванные вероятности и отсутствие сдвига
  p(y|X). Не конкурент, а возможный источник сигнала: при задержке меток driftfdr может
  мониторить оценку CBPE вместо фактической ошибки. Сдвиг p(y|X) CBPE по построению не видит —
  ровно тот случай, который в эксп. 11 оказался вредным.
- **Мониторинг качества NannyML (с метками)** сравнен напрямую в эксп. 17: при сопоставимом
  уровне ложных тревог ловит ухудшения примерно так же; порог ±3σ фиксирован и не учитывает
  число моделей.

## Список литературы

Источники 1–25 — список литературы обзора, в той же нумерации, что и ссылки в тексте.

1. Gama J., Žliobaitė I., Bifet A., Pechenizkiy M., Bouchachia A. A survey on concept drift adaptation // ACM Computing Surveys. 2014. Vol. 46, No. 4. P. 1–37.
2. Cerqueira V., Gomes H. M., Heyden M., Pfahringer B., Bifet A. A framework for evaluating and benchmarking concept drift detection methods // KDD ’26. 2026. arXiv:2606.07789.
3. Page E. S. Continuous inspection schemes // Biometrika. 1954. Vol. 41, No. 1/2. P. 100–115.
4. Gama J., Medas P., Castillo G., Rodrigues P. Learning with drift detection // Brazilian Symposium on Artificial Intelligence. Springer, 2004. P. 286–295.
5. Bifet A., Gavaldà R. Learning from time-changing data with adaptive windowing // Proceedings of the 2007 SIAM International Conference on Data Mining. 2007. P. 443–448.
6. dos Reis D. M., Flach P., Matwin S., Batista G. Fast unsupervised online drift detection using incremental Kolmogorov–Smirnov test // KDD ’16. 2016. P. 1545–1554.
7. Montiel J. [и др.] River: machine learning for streaming data in Python // Journal of Machine Learning Research. 2021. Vol. 22, No. 110. P. 1–8.
8. Bifet A. Classifier concept drift detection and the illusion of progress // International Conference on Artificial Intelligence and Soft Computing. Springer, 2017. P. 715–725.
9. Rombouts J., Wilms I. Monitoring machine learning forecasts for platform data streams. 2024. arXiv:2401.09144.
10. Hu Y. J., Rombouts J., Wilms I. MLOps monitoring at scale for digital platforms. 2025. arXiv:2504.16789.
11. Mei Y. Efficient scalable schemes for monitoring a large number of data streams // Biometrika. 2010. Vol. 97, No. 2. P. 419–433.
12. Chen J., Zhang W., Poor H. V. A false discovery rate oriented approach to parallel sequential change detection problems // IEEE Transactions on Signal Processing. 2020. Vol. 68. P. 1823–1836.
13. Chen Y., Li X. Compound sequential change-point detection in parallel data streams // Statistica Sinica. 2023. Vol. 33, No. 1. P. 453–474.
14. Dandapanthula S., Ramdas A. Multiple testing in multi-stream sequential change detection. 2025. arXiv:2501.04130.
15. Benjamini Y., Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing // Journal of the Royal Statistical Society: Series B. 1995. Vol. 57, No. 1. P. 289–300.
16. Foster D. P., Stine R. A. α-investing: a procedure for sequential control of expected false discoveries // Journal of the Royal Statistical Society: Series B. 2008. Vol. 70, No. 2. P. 429–444.
17. Javanmard A., Montanari A. Online rules for control of false discovery rate and false discovery exceedance // Annals of Statistics. 2018. Vol. 46, No. 2. P. 526–554.
18. Ramdas A., Yang F., Wainwright M. J., Jordan M. I. Online control of the false discovery rate with decaying memory // Advances in Neural Information Processing Systems (NeurIPS). 2017. Vol. 30.
19. Ramdas A., Zrnic T., Wainwright M. J., Jordan M. I. SAFFRON: an adaptive algorithm for online control of the false discovery rate // Proceedings of the 35th International Conference on Machine Learning (ICML). 2018.
20. Gang B., Sun W., Wang W. Structure-adaptive sequential testing for online false discovery rate control // Journal of the American Statistical Association. 2023. Vol. 118, No. 541. P. 732–745.
21. Rebjock Q., Kurt B., Januschowski T., Callot L. Online false discovery rate control for anomaly detection in time series // Advances in Neural Information Processing Systems (NeurIPS). 2021. Vol. 34.
22. Künsch H. R. The jackknife and the bootstrap for general stationary observations // Annals of Statistics. 1989. Vol. 17, No. 3. P. 1217–1241.
23. Politis D. N., Romano J. P. The stationary bootstrap // Journal of the American Statistical Association. 1994. Vol. 89, No. 428. P. 1303–1313.
24. Wu J., Apley D. W. Bootstrapped control limits for score-based concept drift control charts // Technometrics. 2026. arXiv:2507.16749.
25. Souza V. M. A., Reis D. M., Maletzke A. G., Batista G. E. A. P. A. Challenges in benchmarking stream learning algorithms with real-world data // Data Mining and Knowledge Discovery. 2020. Vol. 34. P. 1805–1858.

**Работы вне обзора**, упомянутые в этом документе и в описании метода:

- Podkopaev A., Ramdas A. Tracking the risk of a deployed model and detecting harmful distribution
  shifts // ICLR. 2022.
- Shin J., Ramdas A., Rinaldo A. E-detectors: a nonparametric framework for sequential change
  detection. arXiv:2203.03532 (сверить выходные данные опубликованной версии).
- Wang R., Ramdas A. False discovery rate control with e-values // Journal of the Royal Statistical
  Society: Series B. 2022. Vol. 84, No. 3. P. 822–852.
- Storey J. D., Taylor J. E., Siegmund D. Strong control, conservative point estimation and
  simultaneous conservative consistency of false discovery rates: a unified approach // Journal of
  the Royal Statistical Society: Series B. 2004. Vol. 66, No. 1. P. 187–205.
- Zrnic T., Jiang D., Ramdas A., Jordan M. I. The power of batching in multiple hypothesis testing
  // AISTATS. 2020.
- Bühlmann P. Sieve bootstrap for time series // Bernoulli. 1997. Vol. 3, No. 2. P. 123–148.
- Politis D. N., White H. Automatic block-length selection for the dependent bootstrap //
  Econometric Reviews. 2004. Vol. 23, No. 1. P. 53–70.
- Knijnenburg T. A., Wessels L. F. A., Reinders M. J. T., Shmulevich I. Fewer permutations, more
  accurate P-values // Bioinformatics. 2009. Vol. 25, No. 12. P. i161–i168.
- NannyML: Confidence-based performance estimation (CBPE), документация NannyML,
  https://nannyml.readthedocs.io (статьи-первоисточника нет, метод описан в документации).

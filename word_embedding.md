---
title: "Understanding Word Embedding"
author: "Arjun Paudel"
date: "6/7/2021"
output: 
  html_document: 
    keep_md: yes
editor_options: 
  chunk_output_type: console
---




```r
library(tidyverse)
library(tidytext)
library(textdata)
library(widyr)
```


```r
dt <- tibble(id = c(1,2,1,2),
             text = c("This is a good dog",
                      "Who is a good boy",
                      "Boy, it sure is hot today, huh?",
                      "This is a hot summer month")
             )
dt <- dt %>% arrange(id)
dt
```

```
## # A tibble: 4 x 2
##      id text                           
##   <dbl> <chr>                          
## 1     1 This is a good dog             
## 2     1 Boy, it sure is hot today, huh?
## 3     2 Who is a good boy              
## 4     2 This is a hot summer month
```


```r
tidy_dt <- dt %>% 
  unnest_tokens(word, text)
tidy_dt
```

```
## # A tibble: 23 x 2
##       id word 
##    <dbl> <chr>
##  1     1 this 
##  2     1 is   
##  3     1 a    
##  4     1 good 
##  5     1 dog  
##  6     1 boy  
##  7     1 it   
##  8     1 sure 
##  9     1 is   
## 10     1 hot  
## # ... with 13 more rows
```


```r
tidy_ngram_dt <- dt %>% 
  unnest_tokens(words, text, token = "ngrams", n = 2, collapse = "id") %>% 
  mutate(.ngramID = row_number()) %>% 
  unnest_tokens(word, words) %>% 
  unite(.ngramID, id, .ngramID)

tidy_pmi <- tidy_ngram_dt %>% 
  pairwise_pmi(item = word, feature = .ngramID, sort = TRUE)
tidy_pmi
```

```
## # A tibble: 34 x 3
##    item1  item2    pmi
##    <chr>  <chr>  <dbl>
##  1 huh    today  2.35 
##  2 today  huh    2.35 
##  3 month  summer 2.35 
##  4 summer month  2.35 
##  5 sure   it     1.66 
##  6 it     sure   1.66 
##  7 who    is     0.965
##  8 dog    good   0.965
##  9 good   dog    0.965
## 10 boy    dog    0.965
## # ... with 24 more rows
```

> Notice the use of `collapse = "id"`
>
> Without this, the bigram tokenizer will stop at the end of each row and will not crossover to next row.
>
> With `collapse = "id"` it will cross over to next row with same id and treat last word of previous row and first word of next row together.
>
> In our case, row1 and row3 have same ***id***, so we can think of this as one single long sentence that we just chose to represent in two rows. `collapse = "id"` will treat it such.
>
> Sometimes we read a big text file, certain number of lines at a time. So all the text of same document can end up in different rows. They will share the same document id but not row number. Using `collapse = "document_id"` will treat all those different rows as part of same document.

Implementation from [smltar](https://smltar.com/embeddings.html)


```r
nested_words <- tidy_dt %>%
  nest(words = c(word))

nested_words
```

```
## # A tibble: 2 x 2
##      id words            
##   <dbl> <list>           
## 1     1 <tibble [12 x 1]>
## 2     2 <tibble [11 x 1]>
```

```r
slide_windows <- function(tbl, window_size) {
  skipgrams <- slider::slide(
    tbl, 
    ~.x, 
    .after = window_size - 1, 
    .step = 1, 
    .complete = TRUE
  )
  
  safe_mutate <- safely(mutate)
  
  out <- map2(skipgrams,
              1:length(skipgrams),
              ~ safe_mutate(.x, window_id = .y))

  out %>%
    transpose() %>%
    pluck("result") %>%
    compact() %>%
    bind_rows()
}


tidy_pmi2 <- nested_words %>%
  mutate(words = map(words, slide_windows, 2L)) %>%
  unnest(words) %>%
  unite(window_id, id, window_id) %>%
  pairwise_pmi(word, window_id, sort = TRUE)

tidy_pmi2
```

```
## # A tibble: 34 x 3
##    item1  item2    pmi
##    <chr>  <chr>  <dbl>
##  1 huh    today  2.35 
##  2 today  huh    2.35 
##  3 month  summer 2.35 
##  4 summer month  2.35 
##  5 sure   it     1.66 
##  6 it     sure   1.66 
##  7 who    is     0.965
##  8 dog    good   0.965
##  9 good   dog    0.965
## 10 boy    dog    0.965
## # ... with 24 more rows
```

PMI calculated by both methods is identical.


```r
identical(tidy_pmi, tidy_pmi2)
```

```
## [1] TRUE
```

### Represent words as numeric vectors

Create 5 dimensional vector using `widely_svd()` from **widyr** package


```r
set.seed(13)
tidy_word_vectors <- tidy_pmi %>% 
  widely_svd(
    item1, item2, pmi,
    nv = 5, maxit = 10000
  )
```

Using `irlba()` from **irlba** library


```r
pmi_matrix <- tidy_pmi %>% 
  cast_sparse(item1, item2, pmi)

class(pmi_matrix)
```

```
## [1] "dgCMatrix"
## attr(,"package")
## [1] "Matrix"
```

```r
library(irlba)
```

```
## Loading required package: Matrix
```

```
## 
## Attaching package: 'Matrix'
```

```
## The following objects are masked from 'package:tidyr':
## 
##     expand, pack, unpack
```

```r
set.seed(13)
pmi_svd <- irlba(pmi_matrix, nv=5, maxit = 10000)
pmi_svd
```

```
## $d
## [1] 2.731022 2.729388 2.351375 2.134447 2.031151
## 
## $u
##               [,1]          [,2]          [,3]        [,4]         [,5]
##  [1,]  0.427830391 -0.4286980821  6.603296e-01 -0.02671928  0.008351493
##  [2,]  0.496906697  0.4976166154  2.529126e-01 -0.02425427 -0.007214136
##  [3,]  0.427830391 -0.4286980821 -6.603296e-01 -0.02671928  0.008351493
##  [4,]  0.496906697  0.4976166154 -2.529126e-01 -0.02425427 -0.007214136
##  [5,] -0.015612240  0.0103428563  1.432703e-16 -0.47005832  0.542474331
##  [6,] -0.013734491 -0.0060186571 -8.882720e-16 -0.58136797 -0.655402784
##  [7,] -0.025811206  0.0237291983 -1.586227e-15 -0.06530127  0.026278443
##  [8,] -0.007579327 -0.0006898637 -6.014546e-16 -0.31800405 -0.239164495
##  [9,] -0.009407261  0.0027007791  2.556512e-15 -0.22519094  0.056061119
## [10,] -0.012041000 -0.0007497449  2.543678e-15 -0.47813140  0.447294724
## [11,]  0.363775941 -0.3628283934 -8.227170e-16  0.01145767 -0.005164850
## [12,] -0.073041509 -0.0671095958  7.300637e-16 -0.14442533 -0.055306737
## [13,] -0.017434323  0.0139134335 -2.400901e-15 -0.16322377 -0.107999213
## [14,] -0.026987119 -0.0116183619 -4.007035e-16 -0.07815816 -0.008380762
## 
## $v
##               [,1]          [,2]          [,3]        [,4]         [,5]
##  [1,]  0.496906697 -0.4976166154  6.603296e-01 -0.02425427  0.007214136
##  [2,]  0.427830391  0.4286980821  2.529126e-01 -0.02671928 -0.008351493
##  [3,]  0.496906697 -0.4976166154 -6.603296e-01 -0.02425427  0.007214136
##  [4,]  0.427830391  0.4286980821 -2.529126e-01 -0.02671928 -0.008351493
##  [5,] -0.013734491  0.0060186571 -1.388354e-17 -0.58136797  0.655402784
##  [6,] -0.015612240 -0.0103428563  1.524476e-15 -0.47005832 -0.542474331
##  [7,] -0.073041509  0.0671095958 -3.910444e-15 -0.14442533  0.055306737
##  [8,] -0.009407261 -0.0027007791  3.667094e-15 -0.22519094 -0.056061119
##  [9,] -0.007579327  0.0006898637 -7.394943e-15 -0.31800405  0.239164495
## [10,] -0.012041000  0.0007497449 -5.364881e-15 -0.47813140 -0.447294724
## [11,]  0.363775941  0.3628283934  1.447619e-14  0.01145767  0.005164850
## [12,] -0.025811206 -0.0237291983 -1.175339e-14 -0.06530127 -0.026278443
## [13,] -0.017434323 -0.0139134335  1.988635e-14 -0.16322377  0.107999213
## [14,] -0.026987119  0.0116183619  2.561970e-14 -0.07815816  0.008380762
## 
## $iter
## [1] 2
## 
## $mprod
## [1] 38
```

```r
word_vectors <- pmi_svd$u
rownames(word_vectors) <- rownames(pmi_matrix)
```

Values are almost identical except for third dimension. Not sure if this is rounding error or if these two libraries differ how they implement SVD.  There are also some sign inconsistencies.

> Actually looking at source code of `widely_svd()`, it calls \`irba(). Not really sure what is causing the discrepancy.

> Realized that the inconistencies were due to random nature of the process. Both svd process resulted in identical results after setting same seed before function calls.


```r
word_vectors
```

```
##                [,1]          [,2]          [,3]        [,4]         [,5]
## huh     0.427830391 -0.4286980821  6.603296e-01 -0.02671928  0.008351493
## today   0.496906697  0.4976166154  2.529126e-01 -0.02425427 -0.007214136
## month   0.427830391 -0.4286980821 -6.603296e-01 -0.02671928  0.008351493
## summer  0.496906697  0.4976166154 -2.529126e-01 -0.02425427 -0.007214136
## sure   -0.015612240  0.0103428563  1.432703e-16 -0.47005832  0.542474331
## it     -0.013734491 -0.0060186571 -8.882720e-16 -0.58136797 -0.655402784
## who    -0.025811206  0.0237291983 -1.586227e-15 -0.06530127  0.026278443
## dog    -0.007579327 -0.0006898637 -6.014546e-16 -0.31800405 -0.239164495
## good   -0.009407261  0.0027007791  2.556512e-15 -0.22519094  0.056061119
## boy    -0.012041000 -0.0007497449  2.543678e-15 -0.47813140  0.447294724
## hot     0.363775941 -0.3628283934 -8.227170e-16  0.01145767 -0.005164850
## is     -0.073041509 -0.0671095958  7.300637e-16 -0.14442533 -0.055306737
## this   -0.017434323  0.0139134335 -2.400901e-15 -0.16322377 -0.107999213
## a      -0.026987119 -0.0116183619 -4.007035e-16 -0.07815816 -0.008380762
```

```r
(tidy_word_vectors %>% cast_sparse(item1, dimension, value))
```

```
## 14 x 5 sparse Matrix of class "dgCMatrix"
##                   1             2             3           4            5
## huh     0.427830391 -0.4286980821  6.603296e-01 -0.02671928  0.008351493
## today   0.496906697  0.4976166154  2.529126e-01 -0.02425427 -0.007214136
## month   0.427830391 -0.4286980821 -6.603296e-01 -0.02671928  0.008351493
## summer  0.496906697  0.4976166154 -2.529126e-01 -0.02425427 -0.007214136
## sure   -0.015612240  0.0103428563  1.432703e-16 -0.47005832  0.542474331
## it     -0.013734491 -0.0060186571 -8.882720e-16 -0.58136797 -0.655402784
## who    -0.025811206  0.0237291983 -1.586227e-15 -0.06530127  0.026278443
## dog    -0.007579327 -0.0006898637 -6.014546e-16 -0.31800405 -0.239164495
## good   -0.009407261  0.0027007791  2.556512e-15 -0.22519094  0.056061119
## boy    -0.012041000 -0.0007497449  2.543678e-15 -0.47813140  0.447294724
## hot     0.363775941 -0.3628283934 -8.227170e-16  0.01145767 -0.005164850
## is     -0.073041509 -0.0671095958  7.300637e-16 -0.14442533 -0.055306737
## this   -0.017434323  0.0139134335 -2.400901e-15 -0.16322377 -0.107999213
## a      -0.026987119 -0.0116183619 -4.007035e-16 -0.07815816 -0.008380762
```

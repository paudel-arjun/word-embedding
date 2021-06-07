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

> ***NOTE:*** Notice the use of ***collapse = "id"***
>
> Without this, the bigram tokenizer will stop at the end of each row and will not crossover to next row.
>
> With ***collapse = "id"*** it will cross over to next row with same id and treat last word of previous row and first word of next row together.
>
> In our case, row1 and row3 have same ***id***, so we can think of this as one single long sentence that we just chose to represent in two rows. ***collapse = "id"*** will treat it such.
>
> Sometimes we read a big text file certain number of lines at a time. So all the text of same document can end up in different rows. They will share the same document id but not row number. So using ***collapse = "document_id"*** will treat all those different rows as part of same document.\

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


```r
identical(tidy_pmi, tidy_pmi2)
```

```
## [1] TRUE
```

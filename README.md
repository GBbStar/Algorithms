# 알고리즘 공부 

### 알고리즘 개념
1. 알고리즘이란?
  > 주어진 입력을 출력으로 이끄는 잘 정의된 계산 절차   
  > 계산 문제를 정의하기 위해 입력과 출력의 관계를 잘 정의해야 하는데, 이러한 관계를 구현할 수 있는 계산과정  

2. 타당한 알고리즘
  > 알고리즘이 모든 입력 사례(Instance)에 대해 항상 올바른 출력을 내고, 종료할 경우를 타당하다고 한다  
  > 일부 입력 사례에 종료하지 않거나, 잘못된 답을 도출할 경우 타당하지 않다 일컺는다  

3. 어떤 문제를 알고리즘으로 풀어야할까?
  > 문제들은 공통적으로 "후보는 많지만, 대부분이 문제의 답이 아니다. 최상의 답을 찾는 것을 도전한다", "실용적인 사례를 주변에서 찾을 수 있다"와 같은 특징을 갖는다.
  
4. 알고리즘은 왜 필요할까?
  > 컴퓨터는 유한한 성능을 갖는다. 이로인해 공간적, 시간적 제약이 생긴다.  
  > 이러한 제약사항을 고려하여, 자원을 보다 효율적으로 사용해야 한다. = 효율적인 알고리즘이 필요하다.
  
5. 공부방법
  > 입력과 출력 그리고 사례(인스턴스)를 확실하게 이해한다  
  > 사례를 통해 아이디어(기본적인 이해)를 정리한다  
  > 아이디어를 유사코드로 만들고, 성능을 분석-파악한다  
  > 시간, 공간 복잡도를 이용하여 알고리즘을 효율적으로 최적화한다.  
  
### 삽입정렬
  - 입력 : n개 수들의 수열  (A Sequence of n numbers <a1, a2, ... , an>)
  - 출력 : a1 ≤ a2 ≤ a3 ≤ ... ≤ an을 만족하는 입력 수열의 순열(재배치)   (A Permutation <b1, b2, ... , bn> of the input sequence such that b1 ≤ b2 ≤ ... ≤ bn)
  - 유사코드
  ~~~
  for j = 2 to A.length
      key = A[j]
      i = j-1
      while i>0 and A[i] > key
          A[i+1] = A[i]
          i = i-1
      A[i+1] = key
  ~~~
  - <img src="https://github.com/HwangGyuBin/Algorithms/blob/master/Algorithm%20animation/%EC%82%BD%EC%9E%85%EC%A0%95%EB%A0%AC.gif" width="500" height="300" />
  - 성능 분석하기
  ~~~                                     cost          times
   for j = 2 to A.length                  c₁            n
      key = A[j]                          c₂            n-1
      i = j-1                             c₃            n-1
      while i>0 and A[i] > key            c₄            Σ(j=2 ~ n) tᴊ
          A[i+1] = A[i]                   c₅            Σ(j=2 ~ n) (tᴊ-1)
          i = i-1                         c₆            Σ(j=2 ~ n) (tᴊ-1)
      A[i+1] = key                        c₇            n-1
      
      *) tᴊ는 j가 2, 3, .., n 일때, while 루프의 검사가 실행되는 횟수를 
  ~~~
    

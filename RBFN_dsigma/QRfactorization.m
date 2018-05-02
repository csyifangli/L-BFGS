function [Q1, Q2, R1] = QRfactorization(phi, m, N);

    %QR paragontopoihsh
    [Q,R] = qr(phi);

    %kataskeuth twn Q1, Q2
    Q1 = Q(1:m,1:N);
   % Q2 = Q(1:data,(N+1):data);
    Q2=Q(1:m, N+1:m);

    %Qte = [ Q1 Q2 ]; test

    %kataskeuth tou R1
    R1 = R(1:N,1:N);
USE gold_db;

CREATE TABLE customersNew (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    logo VARCHAR(40),
    email VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
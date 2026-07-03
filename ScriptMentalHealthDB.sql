CREATE TABLE clinica (
	id_clinica INT PRIMARY KEY IDENTITY(1,1),
	nome VARCHAR(50) NOT NULL,
	citta VARCHAR(50) NOT NULL,
	telefono VARCHAR(15) NOT NULL
);
CREATE TABLE ospite (
    id_ospite INT PRIMARY KEY IDENTITY(1,1),
    nome VARCHAR(40) NOT NULL,
    cognome VARCHAR(40) NOT NULL,
    genere VARCHAR(10) NOT NULL,
    eta INT NOT NULL,
    email VARCHAR(40) UNIQUE NOT NULL,
    telefono VARCHAR(15) UNIQUE NOT NULL,
    condizione_mentale VARCHAR(100) CHECK(condizione_mentale IN ('Depression','Normal','Schizophrenia','Stress','Anxiety','Bipolar Disorder')) NOT NULL,
    grave BIT NOT NULL
);
CREATE TABLE camera (
	id_camera INT PRIMARY KEY IDENTITY(1,1),
	numero INT UNIQUE NOT NULL,
	costo DECIMAL (7,2) NOT NULL,
	id_clinica INT NOT NULL,
	FOREIGN KEY (id_clinica) REFERENCES clinica(id_clinica)
	ON DELETE CASCADE 
);
CREATE TABLE staff (
    id_staff INT PRIMARY KEY IDENTITY(1,1),
    nome VARCHAR(40) NOT NULL,
    tipo VARCHAR(30) CHECK(tipo IN('reception', 'manutenzione', 'pulizie','psichiatra','psicologo','infermiere','oss')) NOT NULL,
    compenso DECIMAL (7,2)    NOT NULL
);
CREATE TABLE assegnazione_clienti (
	id_assegnazione_clienti INT PRIMARY KEY IDENTITY(1,1),
	data_inizio DATE NOT NULL,
	data_fine DATE,
	id_ospite INT NOT NULL,
	id_staff INT NOT NULL,
	FOREIGN KEY (id_ospite) REFERENCES ospite(id_ospite) ON DELETE NO ACTION,
	FOREIGN KEY (id_staff) REFERENCES staff(id_staff)	ON DELETE NO ACTION 
);
CREATE TABLE prenotazione_camera (
	id_prenotazione_camera INT PRIMARY KEY IDENTITY(1,1),
	id_camera INT NOT NULL,
	id_ospite INT NOT NULL,
	data_inizio DATE NOT NULL,
	data_fine DATE,
	FOREIGN KEY (id_camera) REFERENCES camera(id_camera) ON DELETE NO ACTION ,
	FOREIGN KEY (id_ospite) REFERENCES ospite(id_ospite) ON DELETE NO ACTION 
);
CREATE TABLE prenotazione_seduta (
	id_prenotazione_seduta INT PRIMARY KEY IDENTITY(1,1),
	id_ospite INT NOT NULL,
	data DATE NOT NULL,
	tipo VARCHAR(50) CHECK(tipo IN ('visita psicologica', 'visita psichiatrica')) NOT NULL,
	prezzo DECIMAL(7,2) NOT NULL,
	FOREIGN KEY (id_ospite) REFERENCES ospite(id_ospite) ON DELETE NO ACTION
);
CREATE TABLE pagamento (
	id_pagamento INT PRIMARY KEY IDENTITY(1,1),
	id_prenotazione_camera INT,
	id_prenotazione_seduta INT,
	tipo VARCHAR(15) NOT NULL CHECK(tipo IN ('contanti', 'carta', 'rate')),
	importo DECIMAL(7,2) NOT NULL,
	FOREIGN KEY (id_prenotazione_camera) REFERENCES prenotazione_camera(id_prenotazione_camera) ON DELETE NO ACTION,
	FOREIGN KEY (id_prenotazione_seduta) REFERENCES prenotazione_seduta(id_prenotazione_seduta) ON DELETE NO ACTION
);
CREATE TABLE medicinale (
	id_medicinale INT PRIMARY KEY IDENTITY(1,1),
	nome VARCHAR(50) NOT NULL
);	

CREATE TABLE prescrizione (
	id_prescrizione INT PRIMARY KEY IDENTITY(1,1),
	id_medicinale INT NULL,
	id_assegnazione_clienti INT NOT NULL,
	quantita VARCHAR(30) NOT NULL,
	FOREIGN KEY (id_medicinale)	REFERENCES medicinale(id_medicinale),
	FOREIGN KEY (id_assegnazione_clienti) REFERENCES assegnazione_clienti(id_assegnazione_clienti)	
);

CREATE TABLE manutenzione (
	id_manutenzione INT PRIMARY KEY IDENTITY(1,1),
	data_inizio DATE NOT NULL,
	data_fine DATE,
	id_camera INT NOT NULL,
	id_staff INT NOT NULL,
	FOREIGN KEY (id_camera) REFERENCES camera(id_camera) ON DELETE NO ACTION,
	FOREIGN KEY (id_staff) REFERENCES staff(id_staff)	ON DELETE NO ACTION 
);
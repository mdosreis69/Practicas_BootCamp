-- Crear la tabla Director
CREATE TABLE director (
    director_id SERIAL PRIMARY KEY,
    name VARCHAR(50),
    surname VARCHAR(50),
    email VARCHAR(50),
    discord_user VARCHAR(25)
);

ALTER TABLE director
ALTER COLUMN name SET NOT NULL,
ALTER COLUMN surname SET NOT NULL,
ALTER COLUMN email SET NOT NULL,
ALTER COLUMN discord_user SET NOT NULL;

-- Crear la tabla Bootcamp
CREATE TABLE bootcamp (
    bootcamp_id SERIAL PRIMARY KEY,
    director_id INT NOT NULL, 
    name VARCHAR(50) NOT NULL,
    start_date DATE NOT NULL,
    duration INT NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    FOREIGN KEY (director_id) REFERENCES director(director_id)
);

-- Crear la tabla Student
CREATE TABLE student (
    student_id SERIAL PRIMARY KEY,
    bootcamp_id INT NOT NULL,
    name VARCHAR(50) NOT NULL,
    surname VARCHAR(50) NOT NULL,
    email VARCHAR(50) NOT NULL,
    phone VARCHAR(100) NOT NULL,
    location VARCHAR(100) NOT NULL,
    FOREIGN KEY (bootcamp_id) REFERENCES bootcamp(bootcamp_id)
);

-- Crea una restriccion UNIQUE en la columna email, para lograr un registro unico en la tabla de estudiantes
ALTER TABLE student
ADD CONSTRAINT unique_email UNIQUE(email);

-- Crear la tabla Teacher
CREATE TABLE teacher (
    teacher_id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    surname VARCHAR(50) NOT NULL,
    email VARCHAR(50) NOT NULL
);

-- Crear la tabla Module
CREATE TABLE module (
    module_id SERIAL PRIMARY KEY,
    bootcamp_id INT NOT NULL,
    teacher_id INT NOT NULL,
    name VARCHAR(50) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
	FOREIGN KEY (bootcamp_id) REFERENCES BOOTCAMP(bootcamp_id),
	FOREIGN KEY (teacher_id) REFERENCES TEACHER(teacher_id)
);

-- Crear un índice en la tabla Module con el campo de fecha de inicio del módulo, para facilitar busquedas posteriores
CREATE INDEX idx_mod_start_date ON MODULE (start_date);

INSERT INTO director (name, surname, email, discord_user) VALUES
('Sandra', 'Navarro', 'sandra.navarro@keepcoding.io', 'sandranavarro'),
('Alberto', 'Casaro', 'alberto.casero@keepcoding.io', 'albertocasero'),
('Xoan', 'Mallon', 'xoan.mallon@keepcoding.io', 'xoanmallon');

INSERT INTO bootcamp (director_id, name, start_date, duration, price) VALUES
('1', 'Big Data, IA & ML',	DATE '2023-11-27', 10, 6000),
('2', 'Desarrollo Web', DATE '2024-06-24', 8, 6000),  
('3', 'Inteligencia Artificial', DATE '2024-06-24', 7	, 5000), 
('1', 'DevOps & Cloud Computing', DATE '2023-06-24', 7, 5500);

INSERT INTO student (bootcamp_id, name, surname, email, phone, location) VALUES
('1', 'Alberto', 'Santos', 'alberto.santos@gmail.com', '+34 693687051', 'España'),
('1', 'Cristina', 'Gomez', 'cristina.gomez@gmail.com', '+34 676686450', 'España'),
('2', 'Lucia', 'Dominguez', 'lucia.dominguez@gmail.com', '+34 784503109','España'),
('2', 'Patricia', 'Meza', 'pmeza@gmail.com', '+52 8035452098', 'Mexico'),
('3', 'Mario', 'Dos Reis', 'mdosreisferreira@gmail.com', '+34 643687451', 'Panama'),
('3', 'Leonardo', 'Britez', 'leo.britez@gmail.com', '+54 91251640714', 'Argentina'),
('4', 'Juan Carlos', 'Avalos', 'jcarlos.avalos@outlook.com', '+54 90771630714', 'Argentina'),
('4', 'Alex', 'Montilla', 'alex.montilla99@gmail.com', '+58 4143391049','Venezuela');

INSERT INTO teacher (name, surname, email) VALUES
('Sandra', 'Navarro', 'sandra.navarro@keepcoding.io'),
('Pablo', 'Alejos',	'pablo.alejos@keepcoding.io'),
('Alex', 'Lopez', 'alex.lopez@keepcoding.io'),
('Tony', 'Bolaño', 'tony.bolano@keepcoding.io'),
('Gonzalo', 'Garcia', 'gonzalo.garcia@keepcoding.io'),
('David', 'Bueno', 'david.bueno@keepcoding.io'),
('Fernando', 'Rodriguez', 'fernando.rodriguez@keepcoding.io'),
('Kevin', 'Martinez', 'kevin.martinez@keepcoding.io'),
('David', 'Jardon', 'david.jardon@keepcoding.io');

INSERT INTO module (bootcamp_id, teacher_id, name, start_date, end_date) VALUES
('1', '1',	'Data 101',	DATE '2023-11-27', DATE '2023-11-29'),
('1', '2',	'Python para BD&ML', DATE '2023-11-30', DATE '2023-12-05'),
('1', '3',	'SQL Avanzado, ETL y Datawarehouse', DATE '2023-12-11',	DATE '2023-12-21'),
('2', '4',	'Desarrollo Web con HTML5 y CSS', DATE '2024-01-15', DATE '2024-01-18'),
('2', '5',	'Programacion PHP y WordPress',	DATE '2024-02-05', DATE '2024-02-08'),
('3', '6',	'Chatbots con Dialogflow', DATE '2024-03-04', DATE '2024-03-08'),
('3', '7',	'Introducción IA y Machine Learning', DATE '2024-03-11', DATE '2024-03-14'),
('4', '8',	'Testing Con Cypress', DATE '2024-04-22', DATE '2024-04-25'),
('4', '9',	'Desarrollar apps iOS con Swift', DATE '2024-05-20', DATE '2024-05-24');


-- Queries de pruebas

-- Encuentra el número total de bootcamp.

SELECT COUNT(*) AS total_bootcamp
	FROM bootcamp;

-- Recupera todos estudiantes inscritos ordenados alfabeticamente por nombre

SELECT name
     , surname
  FROM student
 ORDER BY name ASC;
 
-- Muestra los detalles de los modulos (name, start_date, end_date) para un bootcamp específico (ejemplo bootcamp_id = 1, Big Data, IA & ML).

SELECT name
	, start_date
	, end_date
  FROM module
 WHERE bootcamp_id = 1;

-- Recupera el nombre de un bootcamp por un director específico.

SELECT bootcamp.name
  FROM bootcamp
  JOIN director
    ON bootcamp.director_id = director.director_id
 WHERE bootcamp.director_id = 1;
 
-- Recupera la información del estudiante (student_id, name, surname, email, location) que esta inscrito en un bootcamp con precio superior a 5000

SELECT student_id
	, student.name
	, student.surname
	, student.email
	, location
  FROM student
  JOIN bootcamp
    ON student.bootcamp_id = bootcamp.bootcamp_id
 WHERE bootcamp.price > 5000;
  
-- Obtén el número total de estudiantes en una ubicación específica (España).

SELECT location
     , COUNT(*) AS total_students
  FROM student
 WHERE location = 'España'
 GROUP BY location;
 
-- Encuentra al director con mas bootcamps a su cargo y muestra sus detalles (name, surname, email).

SELECT director.name
     , director.surname
     , director.email
     , max_bootcamp.total_bootcamps
  FROM director
  JOIN (SELECT director_id
             , COUNT(*) AS total_bootcamps
          FROM bootcamp
         GROUP BY director_id
         ORDER BY total_bootcamps DESC
             , director_id
         LIMIT 1) max_bootcamp
    ON director.director_id = max_bootcamp.director_id;

-- Muestra los modulos que empiezan a partir del 2024

SELECT module.name
	, start_date
  FROM module
 WHERE start_date > '2024-01-01';
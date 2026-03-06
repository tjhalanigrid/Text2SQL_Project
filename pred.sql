select count(*) from singer;	concert_singer
select count(*) from singer;	concert_singer
select name, country, age from singer order by age desc;	concert_singer
select name, country, age from singer order by age desc;	concert_singer
select avg (age), min(age) from singer where country = 'france';	concert_singer
select average, min(age), max (age) from singer where country = 'france';	concert_singer
select song_name, song_release_year from singer order by age desc limit 1;	concert_singer
select song_name, song_release_year from singer order by age desc limit 1;	concert_singer
select distinct country from singer where age > 20;	concert_singer
select distinct country from singer where age > 20;	concert_singer
select country, count(*) from singer group by country;	concert_singer
select country, count(*) from singer group by country;	concert_singer
select song_name from singer where age > (select avg(age) from singer);	concert_singer
select song_name from singer where age > (select avg(age) from singer);	concert_singer
select location, name from stadium where capacity between 5000 and 10000;	concert_singer
select location, name fromadium where capacity between 5000 and 10000;	concert_singer
select max(capacity), avg(average) from stadium;	concert_singer
select avg(capacity), max(maximum) from stadium;	concert_singer
select stadium.name, stadium.capacity from stadium join concert on stadium.stadium_id = concert.st stadium_id order by concert.average desc limit 1;	concert_singer
select stadium.name, stadium.capacity from stadium join concert on stadium.stadium_id = concert.st stadium_id order by concert.average desc limit 1;	concert_singer
select count(*) from concert where year = 2014 or 2015;	concert_singer
select count(*) from concert where year > 2014 or year = 2015;	concert_singer
select stadium.name, count(*) from concert join stadium on concert.stadium_id = stadium.stadia_id group by stadium.Stadium_name;	concert_singer
select stadium.name, count(*) from concert join stadium on concert.stadium_id = stadium.stadia_id group by stadium.Stadium_name;	concert_singer
select stadium.name, stadium.capacity from stadium join concert on stadium.stadium_id = concert.st stadium_id where concert.year > 2014 order by count(*) desc limit 1;	concert_singer
select stadium.name, stadium.capacity from stadium join concert on stadium.stadium_id = concert.st stadium_id where concert.year > 2013 group by stadium.Stadium_name order by count(*) desc limit 1;	concert_singer
select year from concert group by year order by count(*) desc limit 1;	concert_singer
select year from concert group by year order by count(*) desc limit 1;	concert_singer
select name from stadium where stadium_id not in (selectadium_id from concert);	concert_singer
select name from stadium where stadium_id not in (selectadium_id from concert);	concert_singer
select country from singer where age > 40 intersect select country from Singer where Age < 30;	concert_singer
select name from stadium except select stadium.name from concert join stadium on concert.stadium_id = stadium.st stadium_id where concert.year =Â 2014;	concert_singer
select name from stadium where stadium_id not in (selectadium_id from concert where year = 2014);	concert_singer
select concert_name, concert_theme from concert group by concert_id;	concert_singer
select concert_name, concert_theme, count(*) from concert;	concert_singer
select singer.name, count(*) from concert join singer_in_concert on concert.concert_id = singer.singer_id group by concert.song_name;	concert_singer
select singer.name, count(*) from concert join singer_in_concert on concert.concert_id = singer.singer_id group by singer.song_name;	concert_singer
select name from singer_in_concert where year = 2014;	concert_singer
select song_name from singer_in_concert where year = 2014;	concert_singer
select name, country from singer where song_name like '% hey%';	concert_singer
select name, country from singer where song_name like '% hey%';	concert_singer
select stadium.name, stadium.location from stadium join concert on stadium.stadium_id = concert.st stadium_id where concert.year >= 2014 intersect select stadium_name,;	concert_singer
select stadium.name, stadium.location from stadium join concert on stadium.stadium_id = concert.st stadium_id where concert.year > 2014 intersect select stadium_name,;	concert_singer
select count(*) from stadium order by capacity desc limit 1;	concert_singer
select count(*) from concert join stadium on concert.stadium_id = stadium.stadia_id where stadium.capacity > (select max(capacity) from stadium);	concert_singer
select count(*) from pets where weight > 10;	pets_1
select count(*) from pets where weight > 10;	pets_1
select weight from pets order by age desc limit 1;	pets_1
select weight from pets order by pet_age desc limit 1;	pets_1
select max(weight), pettype from pets group by pettype;	pets_1
select max(weight), Pettype from pets group by pettype;	pets_1
select count(*) from pets where pet_age > 20;	pets_1
select count(*) from pets where pet_age > 20;	pets_1
select count(*) from pets join has_pet on pets.petid = has_Pet. Petid join student on has_ Pet.stuid =;	pets_1
select count(*) from has_pet where pet_age = 'female';	pets_1
select count(distinct pettype) from pets;	pets_1
select count(distinct pettype) from pets;	pets_1
select fname from student where stuid in (select stuid from has_pet where pettype = 'cat' or pettype = 'dog');	pets_1
select fname from student where stuid in (select stuid from has_pet where pettype = 'cat' or pettype = 'dog');	pets_1
select fname from student where stuid in (select stuid from has_pet where pettype = 'cat' intersect select stuidfrom has_Pet where Pettype = 'dog');	pets_1
select student.fname from student join has_pet on student.stuid = has_Pet.stuid join pets on has_ Pet.petid =;	pets_1
select major, age from student where stuid not in (select stuid from has_pet);	pets_1
select major, age from student where stuid not in (select stuid from has_pet);	pets_1
select stuid from student where stuid not in (select student.stuid from has_pet join petid on has_Pet.petid = petid);	pets_1
select stuid from student where stuid not in (select has_pet.stuid from has_Pet join pets on has_ Pet.petid = pets. Petid where pets.pettype = = 'cat';	pets_1
select fname, age from student where stuid not in (select stuid from has_pet where pettype = 'cat');	pets_1
select fname from student except select student.fname from has_pet join student on has_Pet.stuid = student.stuid where has_ Pet.pettype = = 'dog' and student.sex!= 'cat';	pets_1
select pettype, weight from pets order by pet_age desc limit 1;	pets_1
select pettype, weight from pets order by weight desc limit 1;	pets_1
select petid, weight from pets where pet_age > 1;	pets_1
select petid, weight from pets where pet_age > 1;	pets_1
select avg(pettype), max(petage) from pets group by pettype;	pets_1
select avg(pettype), max(petage) from pets group by pettype;	pets_1
select pettype, avg(weight) from pets group by pettype;	pets_1
select pettype, avg(weight) from pets group by pettype;	pets_1
select fname, age from student where stuid in (select stuid from has_pet);	pets_1
select distinct fname, age from student where stuid in (select stuid from has_pet);	pets_1
select has_pet.petid from has_Pet join student on has_ Pet.stuid = student.stuid where student.lname > 'smith';	pets_1
select petid from has_pet where stuid = 'smith';	pets_1
select count(*), has_pet.petid from has_Pet join student on has_ Pet.stuid = student.stid;	pets_1
select stuid, count(*) from has_pet group by stuid;	pets_1
select fname, sex from student group by fname having count(*) > 1;	pets_1
select fname, sex from student group by fname having count(*) > 1;	pets_1
select lname from student where age = 3;	pets_1
select lname from student where age = 3;	pets_1
select avg(age) from student where stuid not in (select stuid from has_pet);	pets_1
select avg(age) from student where stuid not in (select stuid from has_pet);	pets_1
select count(*) fromcontinents;	car_1
select count(*) fromcontinents;	car_1
select continent, continent fromcontinents group by continent;	car_1
select continent, countryname, count(*) from countries group by continent;	car_1
select count(*) from countries;	car_1
select count(*) from countries;	car_1
select maker, Maker, count(*) from car_makers group by maker;	car_1
select maker, count(*) from car_makers group by maker;	car_1
select model from car_names order by horsepower desc limit 1;	car_1
select model from car_names order by horsepower desc limit 1;	car_1
select model from car_names where weight < (select avg(weight) from cars_data);	car_1
select model from car_names where weight < (select avg(weight) from cars_data);	car_1
select maker from car_makers where year = 1970;	car_1
select distinct maker from car_makers where year = 1970;	car_1
select make, time from car_names order by year desc limit 1;	car_1
select maker, year from car_makers order by year desc limit 1;	car_1
select distinct car_name.model from car_names join model_list on Car_names.model = model_lists.modelid where car_data.year > 1980;	car_1
select count(distinct model) from model_list where year > 1980;	car_1
select continent, count(*) fromcontinents group by continent;	car_1
select continent, count(*) from car_makers group by continent;	car_1
select countryname from car_makers group by country order by count(*) desc limit 1;	car_1
select country from car_makers group by country order by count(*) desc limit 1;	car_1
select count(*), maker from car_makers group by maker;	car_1
select maker, count(*), fullname from car_makers group by maker;	car_1
select accelerate from cars_data where make = 'amc hornet sportabout';	car_1
select sum( Accelerate) from cars_data where make = 'amc hornet sportabout';	car_1
select count(*) from car_makers where make = 'france';	car_1
select count(*) from car_makers where countryname = 'france';	car_1
select count(*) from car_names where make = 'wea';	car_1
select count(*) from car_makers where country = 'united states';	car_1
select avg( MPG) from cars_data where cylinders = 4;	car_1
select avg(mpg) from cars_data where cylinders = 4;	car_1
select min(weight) from cars_data where cylinders = 8;	car_1
select min(weight) from cars_data where year = 1974;	car_1
select maker, model_list.model from car_makers join model_lists on car_manufacturers.manufacturer = model_lists.modelid;	car_1
select maker, model_list.model from car_makers join model_lists on car_manufacturers.manufacturer = model_lists.modelid;	car_1
select countryname, id from car_makers group by country having count(*) >= 1;	car_1
select countryname, countryid from car_makers group by country having count(*) >= 1;	car_1
select count(*) from cars_data where horsepower > 150;	car_1
select count(*) from cars_data where horsepower > 150;	car_1
select avg(weight), avg(year) from cars_data;	car_1
select avg(weight), avg(year) fromcars_data group by year;	car_1
select country from car_makers where country = 'europe' group by country having count(*) >= 3;	car_1
select countryname from car_makers group by country having count(*) >= 3;	car_1
select max(horsepower), make from car_data where cylinders = 3;	car_1
select car_data.horsepower, car_name.make from car_names join cars_data on car_id.make = cars_ data.make where car_ names.cylinders = 3 group by car_lists.model order by count(*) desc limit 3;	car_1
select model from car_data group by model order by max(mpg) desc limit 1;	car_1
select model from car_names order by MPG desc limit 1;	car_1
select avg(horsepower) from cars_data where year < 1980;	car_1
select avg(horsepower) from cars_data where year < 1980;	car_1
select avg(edispl) from cars_data where model = 'volvo';	car_1
select avg(edispl) from cars_data;	car_1
select max(celerate), count(*) from cars_data group by cylinders;	car_1
select max(celerate), count(*) from cars_data group by cylinders;	car_1
select model from car_names group by model order by count(*) desc limit 1;	car_1
select model from model_list group by model order by count(*) desc limit 1;	car_1
select count(*) from cars_data where cylinders > 4;	car_1
select count(*) from cars_data where cylinders > 4;	car_1
select count(*) from cars_data where year = 1980;	car_1
select count(*) from car_makers where make = 1980;	car_1
select count(*) from car_makers where fullname = 'american motor company';	car_1
select count(*) from car_names join model_list on car_makers.manufacturer = model_lists.makeid join car_name on model_names.make =;	car_1
select car_makers.fullname, car_ makers.make from car_manufacturers join model_list on car_ Manufacturers.manufacturer = model_lists.modelid group by car_making.make having count(*) > 3;	car_1
select car_name, make from car_makers group by make having count(*) > 3;	car_1
select distinct model.model from model_list join car_makers on model_lists.manufacturer = car_manufacturers.making where car_names.fullname = = 'general motors' or car_name.weight > 3500;	car_1
select distinct model_list.model from car_names join car_makers on car_name.make = car_players.manufacturer join cars_data on car-names.makeid =;	car_1
select year fromcars_data where weight < 3000 intersect select year from cars_data;	car_1
select count(distinct year) from cars_data where weight < 4000 intersect select year from cars-data where weighted > 3000;	car_1
select horsepower from cars_data order by acceleration desc limit 1;	car_1
select horsepower from cars_data order by acceleration desc limit 1;	car_1
select car_names.cylinders, cars_data.enispl from car_data join car_name on car_id.model = car_ names.model group by car_named.model order by count(*) desc limit 1;	car_1
select sum(cylinders) from car_data where acceleration = (select min(celerate) from cars_data group by model order by count(*) desc limit 1);	car_1
select count(*) from cars_data where horsepower > (select max(horsepower) from car_data);	car_1
select count(*) from cars_data where acceleration > (select max(celerate) from cars-data group by car_id order by horsepower desc limit 1);	car_1
select count(distinct country) from car_makers group by country having count(*) > 2;	car_1
select count(distinct country) from car_makers group by country having count(*) > 2;	car_1
select count(*) from cars_data where cylinders > 6;	car_1
select count(*) from cars_data where cylinders > 6;	car_1
select model.model from car_names join model_list on car_name.model = model_lists.modelid where car_data.horsespl > 4 order by car_ names.horsepower desc limit 4;	car_1
select model from car_names group by model order by horsepower desc limit 4;	car_1
select make, make from car_names where horsepower < 3 group by makeid having count(*) >= 3;	car_1
select make, car_name from cars_data where horsepower < 4;	car_1
select max(mpg) from cars_data where cylinders < 1980;	car_1
select max(mpg) from cars_data where cylinders = 8 or year < 1980;	car_1
select model from car_data where weight > 3500 except select model from model_list where Maker = 'f Ford motor company';	car_1
select distinct model.model from model_list join car_names onmodel_list.model = car_name.model where car_ names.weight > 3500 except select distinct model_lists.model;	car_1
select countryname from countries where countryid not in (select countryid from car_makers);	car_1
select countryname from countries where countryid not in (select countryid from car_makers);	car_1
select id, maker from car_makers group by maker having count(*) >= 2 and car_names.make > 3;	car_1
select make from car_makers group by make having count(*) >= 2 and make > 3;	car_1
select countryid, countryname from car_makers where Maker = 'fiat' group by countryid having count(*) > 3;	car_1
select countryid, countryname from car_makers where Maker = ' fiat' group by countryid having count(*) > 3;	car_1
select country from airports where country = 'jetBlue Airways';	flight_2
select country from airports where country = 'jetblue Airways';	flight_2
select airlines. Abbreviation from airlines join airports on airlines.uid = airports.airline where airports. Airportname > 'jetBlue Airways';	flight_2
select abbreviation from airports where country = 'jetblue Airways';	flight_2
select airline.name, airlines. Abbreviation from airlines join airports on airlines.airline = airports.airlines where airports.country = 'usa';	flight_2
select airlines.airline, airlines.abbreviation from airlines join airports on airlines.uid = airports.airlines where airports.country = 'usa';	flight_2
select airportcode, airportname from airports where city = 'anthony';	flight_2
select airportcode, airportname from airports where city = 'anthony';	flight_2
select count(*) from airlines;	flight_2
select count(*) from airlines;	flight_2
select count(*) from airports;	flight_2
select count(*) from airports;	flight_2
select count(*) from flights;	flight_2
select count(*) from flights;	flight_2
select airline from airlines where abbreviation = 'UAL';	flight_2
select airline from airlines where abbreviation = 'UAL';	flight_2
select count(*) from airlines where country = 'usa';	flight_2
select count(*) from airlines where country = 'usa';	flight_2
select city, country from airports where airportname = 'alton';	flight_2
select city, country from airports where airportname = 'alton';	flight_2
select airportname from airports where airportname = 'ako';	flight_2
select airportname from airports where airportcode = 'ako';	flight_2
select airportname from airports where city = 'aberdeen';	flight_2
select airportname from airports where city = 'aberdeen';	flight_2
select count(*) from flights where airportname = 'apg';	flight_2
select count(*) from flights where airportname = 'apg';	flight_2
select count(*) from flights where destination = 'ato';	flight_2
select count(*) from flights where airportname = 'ato';	flight_2
select count(*) from flights where city = 'aberdeen';	flight_2
select count(*) from flights where city = 'aberdeen';	flight_2
select count(*) from airports where city = 'aberdeen';	flight_2
select count(*) from flights where city = 'aberdeen';	flight_2
select count(*) from flights where city = 'aberdeen' intersect select count(distairport) from flights join airports on flights.airline =;	flight_2
select count(*) from flights where city = 'aberdeen' intersect select count(distairport) from flights as t1 join airports as t2 on t1;	flight_2
select count(*) from flights join airlines on flights.airline = airlines.uid join airports on airlines. Airportcode = 'jetBlue Airways';	flight_2
select count(*) from flights join airlines on flights.airline = airlines.uid join airports on airlines.airname =;	flight_2
select count(*) from flights where airportname = 'united airlines';	flight_2
select count(*) from flights where airportname = 'asy';	flight_2
select count(*) from flights where airportname = 'united airlines';	flight_2
select count(*) from flights where airportname = 'aHD';	flight_2
select count(*) from airports where city = 'aberdeen';	flight_2
select count(*) from flights where city = 'aberdeen';	flight_2
select city from flights group by city order by count(*) desc limit 1;	flight_2
select city from airports group by city order by count(*) desc limit 1;	flight_2
select city from flights group by city order by count(*) desc limit 1;	flight_2
select city from airports group by city order by count(*) desc limit 1;	flight_2
select airportcode from airports group by airportcode order by count(*) desc limit 1;	flight_2
select airportcode from airports group by airportcode order by count(*) desc limit 1;	flight_2
select airportcode from airports group by airportcode order by count(*) desc limit 1;	flight_2
select airportcode from airports group by airportcode order by count(*) desc limit 1;	flight_2
select airline from flights group by airline order by count(*) desc limit 1;	flight_2
select airline from flights group by airline order by count(*) desc limit 1;	flight_2
select airlines.abbreviation, airports.country from airlines join flights on airlines.airline = flights.airlines group by airlines. Abbreviation order by count(*) desc limit 1;	flight_2
select airports.country, flights.airline from airlines join flights on airlines. Airline = flights on flights.Airline group by airports.airname order by count(*) desc limit 1;	flight_2
select airlines.airline from airlines join flights on airlines.uid = flights.uid where airports.airname = 'aHD';	flight_2
select airlines.airline from airlines join flights on airlines.uid = flights. Airline where airports.airportname = 'aHD';	flight_2
select airlines.airline from airlines join flights on airlines.uid = flights.uid where airports.airname = 'aHD';	flight_2
select airlines.airline from airlines join flights on airlines.uid = flights.uid where airports.airname = = 'aHD';	flight_2
select airlines.airline from airlines join flights on airlines.uid = flights.uid where airports.airname = 'apg' intersect select airlines. Airline from airports join flight on airports. Airportcode =;	flight_2
select airlines.airline from airlines join flights on airlines.uid = flights.airlines where airports.airname = = 'apg' intersect select airlines. Airline from flights join airports on flights.Airline =;	flight_2
select airline from flights where airportname = 'cVO' except select airlines.airline from airlines join flights on airlines.uid =;	flight_2
select distinct airlines.airline from airlines join flights on airlines.uid = flights.uid where airports.airname = = 'cvO' except select airlines. Airline from airports join departures on airports. Airportcode =;	flight_2
select airline from flights group by airline having count(*) >= 10;	flight_2
select airline from flights group by airline having count(*) >= 10;	flight_2
select airline from flights group by airline having count(*) < 200;	flight_2
select airline from flights group by airline having count(*) < 200;	flight_2
select flight no from airports where airportname = 'united airlines';	flight_2
select flight no from flights where airlinename = 'united airlines';	flight_2
select flight no from flights where airportname = 'apg';	flight_2
select flight no from flights where airportname = 'apg';	flight_2
select flight no from flights where airportname = 'apg';	flight_2
select flight no from flights where airportname = 'apg';	flight_2
select flightno from flights where city = 'aberdeen';	flight_2
select flight no from flights where city = 'aberdeen';	flight_2
select flightno from flights where city = 'aberdeen';	flight_2
select flight no from flights where city = 'aberdeen';	flight_2
select count(*) from flights join airports on flights.airline = airports.airlines where airports.city =;	flight_2
select count(*) from flights join airports on flights.airline = airports.airlines where airports.city = = 'aberdeen' or airports.country abbrev = 'abilene';	flight_2
select airportname from airports where airportcode not in (select airportcode from flights);	flight_2
select airportname from airports where airportcode not in (select airportcode from flights);	flight_2
select count(*) from employee;	employee_hire_evaluation
select count(*) from employee;	employee_hire_evaluation
select name from employee order by age asc;	employee_hire_evaluation
select name, age from employee order by age asc;	employee_hire_evaluation
select city, count(*) from employee group by city;	employee_hire_evaluation
select count(*), city from employee group by city;	employee_hire_evaluation
select city from employee group by city having count(*) > 1;	employee_hire_evaluation
select city from employee group by city having count(*) > 1;	employee_hire_evaluation
select count(*), location from shop group by location;	employee_hire_evaluation
select location, count(*) from shop group by location;	employee_hire_evaluation
select manager_name, district from shop order by number_products desc limit 1;	employee_hire_evaluation
select manager_name, district from shop order by number_products desc limit 1;	employee_hire_evaluation
select min( Number_products), max(number_product) from shop;	employee_hire_evaluation
select min( Number_products), max(number_product) from shop;	employee_hire_evaluation
select name, location, district from shop order by number_products desc;	employee_hire_evaluation
select name, location, district from shop order by number_products desc;	employee_hire_evaluation
select name from shop where number_products > (select avg(number_products) from shop);	employee_hire_evaluation
select name from shop where number_products > (select avg(number_products) from shop);	employee_hire_evaluation
select name from employee order by year_awarded desc limit 1;	employee_hire_evaluation
select employee.name from employee join employee_hire_evaluation on employee.employee_id = employee. employee_id order by employee.year_awarded desc limit 1;	employee_hire_evaluation
select name from employee order by bonus desc limit 1;	employee_hire_evaluation
select name from employee order by bonus desc limit 1;	employee_hire_evaluation
select name from employee whereemployee_id not in (select employee_id from evaluation);	employee_hire_evaluation
select name from employee whereemployee_id not in (select employee_id from employee_hire_evaluation);	employee_hire_evaluation
select shop.name from hiring join shop on hiring.shop_id = shop.Shop_id order by count(*) desc limit 1;	employee_hire_evaluation
select shop.name from employee join hiring on employee.employee_id = hiring. employee_id group by shop.shop_id order by count(*) desc limit 1;	employee_hire_evaluation
select name from shop except select shop.name from employee join hiring on employee.employee_id = hiring.worker_id;	employee_hire_evaluation
select name from shop except select shop.name from shop join hiring on shop.shop_id = hiring.Shop_id where employee.employee_id not in (select employee_id from employee);	employee_hire_evaluation
select count(*), shop.name from employee join hiring on employee.employee_id = hiring.worker_id group by employee.shop_id;	employee_hire_evaluation
select count(*), shop.name from employee join hiring on employee.employee_id = hiring.employer_id group by shop.shop_id;	employee_hire_evaluation
select sum( Bonus) from evaluation;	employee_hire_evaluation
select sum( Bonus) from evaluation;	employee_hire_evaluation
select * from hiring;	employee_hire_evaluation
select * from hiring;	employee_hire_evaluation
select district from shop where number_products < 3000 intersect select district from store where Number_product > 10000;	employee_hire_evaluation
select district from shop where number_products < 3000 intersect select district fromshop where Number_product > 10000;	employee_hire_evaluation
select count(distinct location) from shop;	employee_hire_evaluation
select count(distinct location) from shop;	employee_hire_evaluation
select count(*) from documents;	cre_Doc_Template_Mgt
select count(*) from documents;	cre_Doc_Template_Mgt
select document_id, document_name, document_description from documents;	cre_Doc_Template_Mgt
select document_id, Document_name, document_description from documents;	cre_Doc_Template_Mgt
select document_name, document_id from documents where document_description like '%w%';	cre_Doc_Template_Mgt
select document_name, template_id from documents where document_description like '%w%';	cre_Doc_Template_Mgt
select document_id, document_description from documents where document_name = 'robbin CV';	cre_Doc_Template_Mgt
select document_id, document_description from documents where document_name = 'robbin cv';	cre_Doc_Template_Mgt
select count(distinct template_type_code) from documents;	cre_Doc_Template_Mgt
select count(distinct template_type_code) from documents;	cre_Doc_Template_Mgt
select count(*) from ref_template_types where template_type_code = 'PPT';	cre_Doc_Template_Mgt
select count(*) from documents where template_type_code = 'pPT';	cre_Doc_Template_Mgt
select templates.template_id, count(*) from documents join ref_template_types on documents.Template_id = ref_Template_types. Template_type_code group by documents.template;	cre_Doc_Template_Mgt
select Template_id, count(*) from documents group by template_id;	cre_Doc_Template_Mgt
select ref_Template_types.template_id, ref_template_Types.Template_type_code from ref_ Template_types join documents on ref_ template_types on documents.template;	cre_Doc_Template_Mgt
select ref_Template_types.template_id, ref_template_types on ref_ Template_types join documents on ref-template_Types.template-id = documents.template;	cre_Doc_Template_Mgt
select Template_id from documents group by template_id having count(*) > 1;	cre_Doc_Template_Mgt
select Template_id from documents group by template_id having count(*) > 1;	cre_Doc_Template_Mgt
select Template_id from documents where document_id not in (select document_ID from documents);	cre_Doc_Template_Mgt
select Template_id from documents except select templates.template_idfrom documents join ref_Template_types on documents.template-id = ref_template_types.template;	cre_Doc_Template_Mgt
select count(*) from ref_template_types;	cre_Doc_Template_Mgt
select count(*) from templates;	cre_Doc_Template_Mgt
select template_id, version_number, template_type_code from ref_template_types;	cre_Doc_Template_Mgt
select template_id, version_number, type_code from templates;	cre_Doc_Template_Mgt
select distinct template_type_code from ref_template_types;	cre_Doc_Template_Mgt
select distinct template_type_code from ref_template_types;	cre_Doc_Template_Mgt
select template_id from ref_template_types where template_type_code = 'ppt' or template_types.template_Type_code = 'pPT';	cre_Doc_Template_Mgt
select Template_id from templates where template_type_code = 'ppt' or template_Type_code like '%pPT%';	cre_Doc_Template_Mgt
select count(*) from ref_template_types where template_type_code = 'cv';	cre_Doc_Template_Mgt
select count(*) from ref_template_types where template_type_code = 'cv';	cre_Doc_Template_Mgt
select ref_template_types.template_type_code, ref_Template_Types.Template_type-code from ref_ template_types join templates on ref_ templates.template-id = templates. Template_id where templates.version_number > 5;	cre_Doc_Template_Mgt
select ref_template_types.template_type_code, ref_Template_types on ref_ templates.Template_id = templates. Template_id where templates.version_number > 5;	cre_Doc_Template_Mgt
select template_type_code, count(*) from ref_template_types group by template_Type_code;	cre_Doc_Template_Mgt
select distinct template_type_code, count(*) from ref_template_types group by template_Type_code;	cre_Doc_Template_Mgt
select template_type_code from ref_template_types group by template_ type_code order by count(*) desc limit 1;	cre_Doc_Template_Mgt
select ref_Template_types.template_type_code from ref_template_types join templates on ref_ template_types!template_Type_code = templates. Template_type;	cre_Doc_Template_Mgt
select template_type_code from ref_template_types group by template_ type_code having count(*) < 3;	cre_Doc_Template_Mgt
select template_type_code from ref_template_types group by template_Type_code having count(*) < 3;	cre_Doc_Template_Mgt
select ref_Template_types.version_number, ref_template_types_type_code from ref_ template_types join templates on ref_ templates.template_id = templates.Template_id group by ref_ Template_Types. Template_Type_code order by count(*) desc limit 1;	cre_Doc_Template_Mgt
select ref_Template_types.template_type_code, ref_template_types on ref_ Template_Types.Template_Type_code from ref_ template_types join templates onRef_Template-types. Template_ type_code = templates.template-type_Code order by ref_ templates.version_number desc limit 1;	cre_Doc_Template_Mgt
select ref_Template_types.template_type_code from ref_template_types join documents on ref_ Template_types!document_id = documents. document_id where documents.document_name like '%data base%';	cre_Doc_Template_Mgt
select ref_Template_types.template_type_code from ref_template_types join documents on ref_ Template_types!document_id = documents. document_id where documents.document_name like '%data base%';	cre_Doc_Template_Mgt
select document_name from documents where template_type_code = 'bk';	cre_Doc_Template_Mgt
select document_name from documents where template_type_code = 'bk';	cre_Doc_Template_Mgt
select template_type_code, count(*) from ref_template_types group by template_Type_code;	cre_Doc_Template_Mgt
select template_type_code, count(*) from ref_template_types group by template_Type_code;	cre_Doc_Template_Mgt
select template_type_code from ref_template_types group by template_Type_code order by count(*) desc limit 1;	cre_Doc_Template_Mgt
select ref_Template_types.template_type_code from ref_template_types join documents on ref_ Template_types!template_Type_code = documents.template-id group by ref_ template_types_code order by count(*) desc limit 1;	cre_Doc_Template_Mgt
select template_type_code from ref_template_types;	cre_Doc_Template_Mgt
select ref_Template_types.template_type_code from ref_template_types join documents on ref_ template_types!template_Type_code = documents.Template_id where documents.document_name!= 'f';	cre_Doc_Template_Mgt
select ref_Template_types.template_type_code, ref_template_types_description from ref_ template_types join templates on ref_ Template_types = templates.Template_Type_code;	cre_Doc_Template_Mgt
select ref_Template_types.template_type_code, ref_template_types on ref_ Template_types.);	cre_Doc_Template_Mgt
select ref_Template_types.template_type_description from ref_template_types join template_types on ref_ template_Types.template-type_code = template_ Types.Template_Type_code where ref_ templates. Template_ type_code like '%ad%';	cre_Doc_Template_Mgt
select ref_Template_types.template_type_description from ref_template_types join ref_ template_types on ref_templates.Template_Type_code = ref_ templates.template-type_code where ref_ Template_Types. Template_type-code like '%ad%';	cre_Doc_Template_Mgt
select ref_Template_types.template_type_code from ref_template_types join template_types on ref_ template_Types.template-type_Code = template_ Types.template;	cre_Doc_Template_Mgt
select ref_Template_types.template_type_code from ref_template_types join template_types on ref_ template_Types.Template_Type_code = template_type-types. Template_type;	cre_Doc_Template_Mgt
select distinct ref_Template_types.template_type_description from ref_template_types join documents on ref_ template_types!document_id = documents. document_id;	cre_Doc_Template_Mgt
select distinct template_description from documents;	cre_Doc_Template_Mgt
select template_id from ref_template_types where template_type_description = 'presentation';	cre_Doc_Template_Mgt
select Template_id from templates where template_description = 'presentation';	cre_Doc_Template_Mgt
select count(*) from paragraphs;	cre_Doc_Template_Mgt
select count(*) from paragraphs;	cre_Doc_Template_Mgt
select count(*) from documents where document_name = 'summer show';	cre_Doc_Template_Mgt
select count(*) from documents where document_name = 'summer show';	cre_Doc_Template_Mgt
select paragraph_details from paragraphs where paragraph_text = 'korea';	cre_Doc_Template_Mgt
select other_details from paragraphs where paragraph_text like '%korea%';	cre_Doc_Template_Mgt
select document_id, document_text from documents where document_name = 'Welcome to NY';	cre_Doc_Template_Mgt
select document_id, document_text from documents where document_name = 'Welcome to NY';	cre_Doc_Template_Mgt
select paragraph_text from documents where document_name = 'customer reviews';	cre_Doc_Template_Mgt
select paragraph_text from documents where document_name = 'customer reviews';	cre_Doc_Template_Mgt
select document_id, count(*) from documents group by documents_id;	cre_Doc_Template_Mgt
select document_id, count(*) from documents group by document_ID;	cre_Doc_Template_Mgt
select document_id, document_name, count(*) from documents;	cre_Doc_Template_Mgt
select document_id, document_name, count(*) from documents;	cre_Doc_Template_Mgt
select document_id from documents group by document_ID having count(*) >= 2;	cre_Doc_Template_Mgt
select document_id from documents group by document_ID having count(*) >= 2;	cre_Doc_Template_Mgt
select document_id, document_name from documents order by count(*) desc limit 1;	cre_Doc_Template_Mgt
select document_id, document_name from documents order by paragraph_id desc limit 1;	cre_Doc_Template_Mgt
select document_id from documents order by paragraphs desc limit 1;	cre_Doc_Template_Mgt
select document_id from documents order by paragraph_id desc limit 1;	cre_Doc_Template_Mgt
select document_id from documents where paragraph_id between 1 to 2;	cre_Doc_Template_Mgt
select document_id from documents where paragraph_text between 1 and 2;	cre_Doc_Template_Mgt
select document_id from documents where document_name = 'brazil' intersect select document_ID from documents join paragraphs on documents.paragraph_id =;	cre_Doc_Template_Mgt
select document_id from documents where document_name = 'brazil' intersect select document_ID from documents joinagraphs on documents.Paragraph_id =agraphs.paragraphs_id where documents. Paragraph_text = 'ryland' and documents.document_name = 'Ireland';	cre_Doc_Template_Mgt
select count(*) from teacher;	course_teach
select count(*) from teacher;	course_teach
select name from teacher order by age asc;	course_teach
select name from teacher order by age asc;	course_teach
select age, hometown from teacher;	course_teach
select age, hometown from teacher;	course_teach
select name from teacher where hometown!= 'little lever urban district';	course_teach
select name from teacher where hometown!= 'little lever urban district';	course_teach
select name from teacher where age > 32 or age = 33;	course_teach
select name from teacher where age > 32 or age = 33;	course_teach
select hometown from teacher order by age desc limit 1;	course_teach
select name from teacher order by age desc limit 1;	course_teach
select hometown, count(*) from teacher group by hometown;	course_teach
select hometown, count(*) from teacher group by hometown;	course_teach
select hometown from teacher group by hometown order by count(*) desc limit 1;	course_teach
select hometown from teacher group by hometown order by count(*) desc limit 1;	course_teach
select hometown from teacher group by hometown having count(*) >= 2;	course_teach
select town from teacher group by town having count(*) >= 2;	course_teach
select teacher.name, course.course_id from course_arrange join teacher on course.teacher_id = teacher.teachers_id;	course_teach
select teacher.name, course.course_id from course join teacher on course.teacher_id = teacher.teaching_id;	course_teach
select name, course.course_id from course join teacher on course.teacher_id = teacher.teaching_id order by teacher.name;	course_teach
select teacher.name, course.course_id from course join teacher on course_arrange.teacher_id = teacher.teachers_id order by course. Course_id asc;	course_teach
select t1.name from course_arrange as t1 join teacher as t2 on t2.teacher_id = t3.teachers_id join course_ARRange as T4 on t3 on t4.course_id;	course_teach
select name from course_arrange where grade = 'math';	course_teach
select name, count(*) from teacher group by name;	course_teach
select name, count(*) from teacher group by name;	course_teach
select name from teacher group by name having count(*) >= 2;	course_teach
select name from teacher group by name having count(*) >= 2;	course_teach
select name from teacher whereteacher_id not in (select teacher_id from course_arrange);	course_teach
select name from teacher whereteacher_id not in (select teacher_id from course_arrange);	course_teach
select count(*) from visitor where age < 30;	museum_visit
select visitor.name from visitor join museum on visitor.id = museum.museum_id where visitor.level_of_membership > 4 order by visitor. Level_of-membership desc;	museum_visit
select avg(age) from visitor where level_of_membership < 4;	museum_visit
select name, level_of_membership from visitor where age > 4 order by age desc;	museum_visit
select museum.id, museum.name from museum join visitor on museum.museum_id = visitor.id group by visitor.name order by count(*) desc limit 1;	museum_visit
select avg(num_of_staff) from museum where open_year < 2009;	museum_visit
select open_year, num_of_staff from museum where name = 'plaza museum';	museum_visit
select name from museum where open_year > 2010;	museum_visit
select museum.id, museum.name, visitor.age from museum joinvisit on museum.museum_id = visit.id group by visitor.id having count(*) > 1;	museum_visit
select museum.id, museum.name, visitor.level_of_membership from museum join visitor on museum.museum_id = visitor_id group by visitor.id order by sum(visit.total_spent) desc limit 1;	museum_visit
select museum.id, museum.name from museum join visitor on museum.museum_id = visitor.id order by visitor.total_spent desc limit 1;	museum_visit
select name from museum whereuseum_id not in (select museum_id from visitor);	museum_visit
select visitor.name, visitor.age from visitor join museum on visitor.id = museum.museum_id order by museum.total_spent desc limit 1;	museum_visit
select avg(num_of_ticket), max(total_spent) from visit;	museum_visit
select sum(total_spent) from visitor where level_of_membership = 1;	museum_visit
select visitor.name from museum join visitor on museum.museum_id = visitor.id where museum.open_year < 2009 intersect select visitor. Name from museum_visit joinvisit on museum_ VISIT.visit_id =visit.visitor_id where visitor. Open_Year > 2011;	museum_visit
select count(*) from museum where open_year < 2010;	museum_visit
select count(*) from museum where open_year > 2013 or open_ year < 2008;	museum_visit

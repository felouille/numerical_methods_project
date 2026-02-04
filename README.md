In the main.py file there are all the functions used in the project. As the file may chaotic to navigate, here are some guidelines:
  - In the advection-only equations, the celerity must be a constant
  - In the other equations, the celerity should be a field (just as the velocity)
  - Please be aware that these rule may not always be respected in some functions, as I may have used a field-like celerity for convenience (while not updating the field in the function). Please be careful and have a look inside the function before testing it.
  - There are a lot of comments, as this project was done bit by bit. Some functions were used to illustrate the first points of the subjects, only to never be used again. Because of this, the file is huge and may be confusing.
  - The entire file only requires numpy and matplotlib.

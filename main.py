import inquirer
from vasijas import vasijas
from glifos import glifos
from glifos import variantes

def main():
    # input_dir = 'assets/dataset_glifos/a'
    # output_dir = 'assets/dataset_glifos/generated_images'
    # variantes.process_directory(input_dir, output_dir, 50)

    questions = [
        inquirer.List('opcion',
                      message="¿Qué desea realizar?",
                      choices=['1. Análisis de glifos',
                               '2. Reconocimiento de vasijas',
                               '3. Reconocimiento de terrenos',
                               '4. Salir'],
                      ),
    ]

    answers = inquirer.prompt(questions)

    if answers['opcion'] == '1. Análisis de glifos':
        print("Ejecutando análisis de glifos")
        # glifos.train()
        glifos.predict()
    elif answers['opcion'] == '2. Reconocimiento de vasijas':
        print("Ejecutando reconocimiento de vasijas")
        vasijas.main()
    elif answers['opcion'] == '3. Reconocimiento de terrenos':
        print("Ejecutando reconocimiento de terrenos")
        # terrenos()
    elif answers['opcion'] == '4. Salir':
        print("Saliendo del programa")
        exit()

    else:
        print("Opción incorrecta")
        main()

if __name__ == "__main__":
    main()
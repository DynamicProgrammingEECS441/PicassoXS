//
//  ViewController.swift
//  PicassoXs
//
//  Created by Senhao Wang on 3/19/20.
//  Copyright © 2020 Senhao Wang. All rights reserved.
//

import UIKit
import AVFoundation


struct GlobalVariables {
    static var curInput: UIImage?
    static var curFilter: UIImage?
    static var curResult: UIImage?
    static var FilteredHistory: [MadeImages] = []
    static var Filters: [Filter] = []
}



class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate,UITableViewDelegate, UITableViewDataSource{
    
    
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return GlobalVariables.FilteredHistory.count
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let curHistory = GlobalVariables.FilteredHistory[indexPath.row]
        let cell = tableView.dequeueReusableCell(withIdentifier: "HomeViewCell") as! HomeViewCell
        cell.setHomeView(Image: curHistory)
        
        return cell
    }
    

    //Screen
        override func viewDidLoad() {
            super.viewDidLoad()
            self.loadFilterLibrary()
            let im1 = MadeImages(UIImage(named:"I1")!, UIImage(named:"S6")!, UIImage(named: "R5")!)
            let im2 = MadeImages(UIImage(named:"I4")!, UIImage(named:"S2")!, UIImage(named: "R4")!)
            let im3 = MadeImages(UIImage(named:"I4")!, UIImage(named:"S2")!, UIImage(named: "R2")!)
            let im4 = MadeImages(UIImage(named:"I10")!, UIImage(named:"S6")!, UIImage(named: "R6")!)

            GlobalVariables.FilteredHistory.append(im1)
            GlobalVariables.FilteredHistory.append(im2)
            GlobalVariables.FilteredHistory.append(im3)
            GlobalVariables.FilteredHistory.append(im4)
            
    }
    
    
    
    
    
    
    
    
    
    //使用照相机按键
    @IBAction func Camera(_ sender: Any) {
        print("camera Tapped")
        
        if UIImagePickerController.isSourceTypeAvailable(.camera) {
            let picker = UIImagePickerController()
                picker.delegate = self
            picker.sourceType = UIImagePickerController.SourceType.camera
                picker.allowsEditing = false

            picker.mediaTypes = UIImagePickerController.availableMediaTypes(for:.camera)!
            print(picker.mediaTypes)
            self.present(picker, animated: true, completion: nil)
            }
        else {
                print("can't find camera")
            }
        }
    
    //使用Library 按键
    @IBAction func Library(_ sender: Any) {
        print("Library Tapped")
        if UIImagePickerController.isSourceTypeAvailable(UIImagePickerController.SourceType.photoLibrary) {
          let imagePicker = UIImagePickerController()
          imagePicker.delegate = self
            imagePicker.sourceType = UIImagePickerController.SourceType.photoLibrary
          imagePicker.allowsEditing = false
          self.present(imagePicker, animated: true, completion: nil)
            
        }
    }
    
    
    //用户取消拍照
     func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
         dismiss(animated: true, completion: nil)
     }
    
    //用户拍照完成
     func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
         print("\(info)")
         if let image = info[.originalImage] as? UIImage {
            
            //Set Current Input Image
            GlobalVariables.curInput=image
            dismiss(animated: true, completion: nil)
            performSegue(withIdentifier: "Filter", sender: nil)
         }
     }
    
   
    
    
    func loadFilterLibrary(){
        
        let S0 = UIImage(named: "S0")
        GlobalVariables.Filters.append(Filter(S0!,"Add Your Own", "ADDDDDDDYOUROWN"))
        
        let S1 = UIImage(named: "S1")
        GlobalVariables.Filters.append(Filter(S1!,"Pablo Picasso", "Pablo Ruiz Picass, used color as an expressive element, but relied on drawing rather than subtleties of color to create form and space. Picasso displayed an interest in subject matter of every kind and demonstrated a great stylistic versatility that enabled him to work in several styles at once."))
        
        let S2 = UIImage(named: "S21")
        GlobalVariables.Filters.append(Filter(S2!,"Pop Art", "Pop art is an art movement that emerged in the United Kingdom and the United States during the mid- to late-1950s.[1][2] The movement presented a challenge to traditions of fine art by including imagery from popular and mass culture, such as advertising, comic books and mundane mass-produced cultural objects."))
        
        let S3 = UIImage(named: "S16")
        GlobalVariables.Filters.append(Filter(S3!,"Salvador Dalí", "Salvador Dalí, Born in Figueres, Catalonia, Dalí was a skilled artist draftsman, best known for the striking and bizarre images in his work. His painterly skills are often attributed to the influence of Renaissance masters."))
        
        let S4 = UIImage(named: "S6")
        GlobalVariables.Filters.append(Filter(S4!,"Giorgio Morandi", "Giorgio Morandi (July 20, 1890 – June 18, 1964) was an Italian painter and printmaker who specialized in still life. His paintings are noted for their tonal subtlety in depicting apparently simple subjects, which were limited mainly to vases, bottles, bowls, flowers and landscapes." ))
        
        let S5 = UIImage(named: "S5")
        GlobalVariables.Filters.append(Filter(S5!,"Claude Monet", "Oscar-Claude Monet (November 1840 – 5 December 1926) was a French painter, a founder of French Impressionist painting and the most consistent and prolific practitioner of the movement's philosophy of expressing one's perceptions before nature, especially as applied to plein air landscape painting."))
        
        let S6 = UIImage(named: "S9")
        GlobalVariables.Filters.append(Filter(S6!,"Vincent van Gogh","Vincent Willem van Gogh (30 March 1853 – 29 July 1890) was a Dutch post-impressionist painter who is among the most famous and influential figures in the history of Western art. In just over a decade, he created about 2,100 artworks, including around 860 oil paintings, most of which date from the last two years of his life. "))
        
        let S7 = UIImage(named: "S17")
        GlobalVariables.Filters.append(Filter(S7!,"Leonardo da Vinci","Leonardo di ser Piero da Vinci, was an Italian polymath of the Renaissance whose areas of interest included invention, drawing, painting, sculpture, architecture, science, music, mathematics, engineering, literature, anatomy, geology, astronomy, botany, paleontology, and cartography. He has been variously called the father of palaeontology, ichnology, and architecture, and is widely considered one of the greatest painters of all time"))
        
        let S8 = UIImage(named: "S20")
        GlobalVariables.Filters.append(Filter(S8!,"Ukiyo-e", "Ukiyo-e is a genre of Japanese art which flourished from the 17th through 19th centuries. Its artists produced woodblock prints and paintings of such subjects as female beauties; kabuki actors and sumo wrestlers; scenes from history and folk tales; travel scenes and landscapes; flora and fauna; and erotica. "))
        
    }
    
}



